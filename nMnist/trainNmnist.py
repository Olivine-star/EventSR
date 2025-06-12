import sys
import os
import datetime
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../')
from model import NetworkBasic
from nMnist.mnistDatasetSR import mnistDataset
from utils.ckpt import checkpoint_restore, checkpoint_save
from opts import parser
from statistic import Metric
import slayerSNN as snn




def main():
    args = parser.parse_args()
    shape = [34, 34, 350]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = 'cuda'

    trainDataset = mnistDataset()
    testDataset = mnistDataset(False)

    print("Training sample: %d, Testing sample: %d" % (len(trainDataset), len(testDataset)))
    bs = args.bs

    trainLoader = DataLoader(dataset=trainDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=bs, shuffle=True, num_workers=args.j, drop_last=False)

    netParams = snn.params('network.yaml')
    m = NetworkBasic(netParams)
    m = torch.nn.DataParallel(m).to(device)
    print(m)

    MSE = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr, amsgrad=True)

    iter_per_epoch = len(trainDataset) // bs
    time_last = datetime.datetime.now()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    savePath = os.path.join(
        args.savepath,
        f"bs{args.bs}_lr{args.lr}_ep{args.epoch}_cuda{args.cuda}_{timestamp}"
    )

    os.makedirs(savePath, exist_ok=True)

    m, epoch0 = checkpoint_restore(m, savePath)

    maxEpoch = args.epoch
    showFreq = args.showFreq
    valLossHistory = []
    tf_writer = SummaryWriter(log_dir=savePath)

    with open(os.path.join(savePath, 'config.txt'), 'w') as f:
        for i, config in enumerate(m.module.neuron_config):
            f.writelines('layer%d: theta=%d, tauSr=%.2f, tauRef=%.2f, scaleRef=%.2f, tauRho=%.2f, scaleRho=%.2f\n' % (
                i + 1, config['theta'], config['tauSr'], config['tauRef'], config['scaleRef'], config['tauRho'], config['scaleRho']))
        f.writelines('\n')
        f.write(str(args))

    log_training = open(os.path.join(savePath, 'log.csv'), 'w')

    for epoch in range(epoch0 + 1, maxEpoch):
        trainMetirc = Metric()
        m.train()
        for i, (eventLr, eventHr) in enumerate(trainLoader, 0):
            eventLr, eventHr = eventLr.to(device), eventHr.to(device)
            output = m(eventLr)

            loss = MSE(output, eventHr)
            loss_ecm = sum([MSE(torch.sum(output[:, :, :, :, i*50:(i+1)*50], dim=4),
                                torch.sum(eventHr[:, :, :, :, i*50:(i+1)*50], dim=4)) for i in range(shape[2] // 50)])
            loss_total = loss + loss_ecm * 5

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if i % showFreq == 0:
                trainMetirc.updateIter(loss.item(), loss_ecm.item(), loss_total.item(), 1,
                                       eventLr.sum().item(), output.sum().item(), eventHr.sum().item())
                print_progress(epoch, maxEpoch, i, iter_per_epoch, bs, trainMetirc, time_last, "Train", log_training)
                time_last = datetime.datetime.now()

        log_tensorboard(tf_writer, trainMetirc, epoch, prefix="Train")
        log_epoch_done(log_training, epoch)

        if epoch % 1 == 0:
            m.eval()
            t = datetime.datetime.now()
            valMetirc = Metric()
            for i, (eventLr, eventHr) in enumerate(testLoader, 0):
                with torch.no_grad():
                    eventLr, eventHr = eventLr.to(device), eventHr.to(device)
                    output = m(eventLr)

                    loss = MSE(output, eventHr)
                    loss_ecm = sum([MSE(torch.sum(output[:, :, :, :, i*50:(i+1)*50], dim=4),
                                        torch.sum(eventHr[:, :, :, :, i*50:(i+1)*50], dim=4)) for i in range(shape[2] // 50)])
                    loss_total = loss + loss_ecm
                    valMetirc.updateIter(loss.item(), loss_ecm.item(), loss_total.item(), 1,
                                         eventLr.sum().item(), output.sum().item(), eventHr.sum().item())

                    if i % showFreq == 0:
                        print_progress(epoch, maxEpoch, i, len(testDataset) // bs, bs, valMetirc, time_last, "Val", log_training)
                        time_last = datetime.datetime.now()

            log_tensorboard(tf_writer, valMetirc, epoch, prefix="Val")
            log_validation_summary(valMetirc, valLossHistory, epoch, t, log_training, savePath, m, device)

        if (epoch + 1) % 15 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                print("Learning rate decreased to:", param_group['lr'])

def print_progress(epoch, maxEpoch, i, total, bs, metric, time_last, mode, log_file):
    remainIter = (maxEpoch - epoch - 1) * total + (total - i - 1)
    now = datetime.datetime.now()
    dt = (now - time_last).total_seconds()
    remainSec = remainIter * dt
    h, remain = divmod(remainSec, 3600)
    m, s = divmod(remain, 60)
    end_time = now + datetime.timedelta(seconds=remainSec)
    avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = metric.getAvg()
    msg = f'{mode}, Cost {dt:.1f}s, Epoch[{epoch}], Iter {i}/{total}, Time Loss: {avgLossTime:.6f}, ' \
          f'Ecm Loss: {avgLossEcm:.6f}, Avg Loss: {avgLoss:.6f}, bs: {bs}, IS: {avgIS}, OS: {avgOS}, GS: {avgGS}, ' \
          f'Remain time: {int(h):02d}:{int(m):02d}:{int(s):02d}, End at: {end_time:%Y-%m-%d %H:%M:%S}'
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()

def log_tensorboard(writer, metric, epoch, prefix="Train"):
    avgLossTime, avgLossEcm, avgLoss, avgIS, avgOS, avgGS = metric.getAvg()
    writer.add_scalar(f'loss/{prefix}_Time_Loss', avgLossTime, epoch)
    writer.add_scalar(f'loss/{prefix}_Spatial_Loss', avgLossEcm, epoch)
    writer.add_scalar(f'loss/{prefix}_Total_Loss', avgLoss, epoch)
    writer.add_scalar(f'SpikeNum/{prefix}_Input', avgIS, epoch)
    writer.add_scalar(f'SpikeNum/{prefix}_Output', avgOS, epoch)
    writer.add_scalar(f'SpikeNum/{prefix}_GT', avgGS, epoch)

def log_epoch_done(log_file, epoch):
    msg = '-' * 50 + f"Epoch {epoch} Done" + '-' * 50
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()

def log_validation_summary(metric, valLossHistory, epoch, t_start, log_file, savePath, model, device):
    avgLossTime, avgLossEcm, avgLoss, *_ = metric.getAvg()
    valLossHistory.append(avgLoss)
    t_end = datetime.datetime.now()
    msg = f"Validation Done! Cost Time: {(t_end - t_start).total_seconds():.2f}s, " \
          f"Loss Time: {avgLossTime:.6f}, Loss Ecm: {avgLossEcm:.6f}, Avg Loss: {avgLoss:.6f}, " \
          f"Min Val Loss: {min(valLossHistory):.6f}\n"
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()

    checkpoint_save(model=model, path=savePath, epoch=epoch, name="ckpt", device=device)
    if avgLoss == min(valLossHistory):
        checkpoint_save(model=model, path=savePath, epoch=epoch, name="ckptBest", device=device)
    with open(os.path.join(savePath, 'log.txt'), "a") as f:
        f.write(f"Epoch: {epoch}, Ecm loss: {avgLossEcm:.6f}, Spike time loss: {avgLossTime:.6f}, Total loss: {avgLoss:.6f}\n")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
