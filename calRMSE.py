import numpy as np
import os
import math


def calRMSE(eventOutput, eventGt):
    xOp = np.round(eventOutput[:, 1]).astype(int)
    yOp = np.round(eventOutput[:, 2]).astype(int)
    pOp = np.round(eventOutput[:, 3]).astype(int)
    tOp = np.round(eventOutput[:, 0]).astype(int)

    xGt = np.round(eventGt[:, 1]).astype(int)
    yGt = np.round(eventGt[:, 2]).astype(int)
    pGt = np.round(eventGt[:, 3]).astype(int)
    tGt = np.round(eventGt[:, 0]).astype(int)

    VoxOp = np.zeros([2, _H, _W, _T])
    VoxOp[pOp, xOp, yOp, tOp] = 1
    VoxGt = np.zeros([2, _H, _W, _T])
    VoxGt[pGt, xGt, yGt, tGt] = 1
    ecm = np.sum(np.sum(VoxGt, axis=3), axis=0)
    assert ecm.sum() == xGt.shape[0]

    RMSE1 = np.sum( (VoxGt - VoxOp) * (VoxGt - VoxOp) )
    RMSE2 = 0
    for k in range(math.ceil(_T/50)):
        psthGt = np.sum(VoxGt[:, :, :, k*50:(k+1)*50], axis=3)
        psthOp = np.sum(VoxOp[:, :, :, k*50:(k+1)*50], axis=3)
        RMSE2 += np.sum( (psthGt - psthOp) * (psthGt - psthOp) )

    RMSE = np.sqrt( (RMSE1 + RMSE2) / ( (tGt.max()-tGt.min()) * np.sum(ecm!=0) ))

    return RMSE, np.sqrt( (RMSE1) / ( (tGt.max()-tGt.min()) * np.sum(ecm!=0) )), np.sqrt( (RMSE2) / ( (tGt.max()-tGt.min()) * np.sum(ecm!=0) ))



# path = "./dataset/N-MNIST/SR_Test"
# path1 = "./dataset/N-MNIST/SR_Test"
# _H, _W, _T = [34, 34, 350]

# path = "./dataset/Cifar10-DVS/SR_Test"
# path1 = "./dataset/Cifar10-DVS/ResConv/HRPre"
# _H, _W, _T = [128, 128, 1500]

# path = "./dataset/asl/SR_Test"
# path1 = "./dataset/asl/ResConv/HRPre"
# _H, _W, _T = [240, 180, 600]

# path = "./dataset/ImageReconstruction/SR_Test"
# path1 = "./dataset/ImageReconstruction/ResConv/HRPre"
# _H, _W, _T = [240, 180, 600]

def load_path_config(path_config='dataset_path.txt'):
    path_dict = {}
    with open(path_config, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                path_dict[key.strip()] = val.strip()
    return path_dict


# path = "D:/PycharmProjects/EventSR-dataset/dataset/N-MNIST/SR_Test"
# path1 = "D:/PycharmProjects/EventSR-dataset/dataset/N-MNIST/ResConv/HRPre"
# _H, _W, _T = [240, 180, 600]


paths = load_path_config()
path1 = paths.get('savepath', '')           # Output
path2 = paths.get('sr_test_root', '')       # Gt


_H, _W, _T = [240, 180, 600]

# 获取path2目录下HR文件夹中的所有文件名
# 读取 Ground Truth 高分辨率事件数据路径下的所有类别名目录（例如 N-MNIST 的 0 到 9 类），
# 后续会依次对每个类别目录中的样本进行 RMSE 评估。
classList = os.listdir(os.path.join(path2, 'HR'))

RMSEListOurs, RMSEListOurs_s, RMSEListOurs_t = [], [], []


i = 1
for n in classList:
    print(n)
    p1 = os.path.join(path1, n)              # Output
    p2 = os.path.join(path2, 'HR', n)                 # Gt

    k = 1
    sampleList = os.listdir(p2)

    for name in sampleList:
        eventOutput = np.load(os.path.join(p1, name))
        eventGt = np.load(os.path.join(p2, name))

        RMSE, RMSE_t, RMSE_s = calRMSE(eventOutput, eventGt)
        RMSEListOurs.append(RMSE)
        RMSEListOurs_s.append(RMSE_s)
        RMSEListOurs_t.append(RMSE_t)

        print(i, '/', len(classList), '  ', k, '/', len(sampleList), RMSE)
        k += 1
    i += 1

print(sum(RMSEListOurs) / len(RMSEListOurs))

#1
with open(path1 + '/result.txt', 'w') as f:
    f.writelines('Ours RMSE: ' + str(sum(RMSEListOurs)/len(RMSEListOurs)) + ', Ours RMSE_s: ' + str(sum(RMSEListOurs_s)/ len(RMSEListOurs)) + ', Ours RMSE_t: ' + str(sum(RMSEListOurs_t)/ len(RMSEListOurs)) + '\n')