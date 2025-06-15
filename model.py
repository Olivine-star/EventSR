import torch
import slayerSNN as snn
from utils.utils import getNeuronConfig
import numpy as np


class NetworkBasic(torch.nn.Module):
    """
    这个模型就由trainNmnist传入两个参数：netParams和传入spikeInput
    netParams = snn.params('network.yaml'),调用模型类，创建网络对象,m = NetworkBasic(netParams)
    output = m(eventLr)，eventLr就是传入forward的参数spikeInput
    """
    #
    def __init__(self, netParams,
                 theta=[30, 50, 100],
                 tauSr=[1, 2, 4],
                 tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1],
                 tauRho=[1, 1, 10],
                 scaleRho=[10, 10, 100]):
        super(NetworkBasic, self).__init__()

        """
        每一层神经元的行为由 getNeuronConfig(...) 函数配置，如阈值、衰减时间等，是 SNN 特有的。
        这是为了给每一层设置不同的神经元行为参数，因为在 SNN 中：神经元不再是普通的 ReLU 激活单元，而是具有生物启发特性的脉冲神经元；

        参数如 theta（发放阈值）、tauSr（突触响应时间常数）、tauRef（不应期时间）、scaleRho（电压或重置幅值）等，会影响神经元的发放频率、时间响应、抑制机制等。

        ✅ 目的：通过为不同层设置不同神经元行为，让网络在不同阶段能学习到更合适的时间动态。
        
        """
        # 依次将三个不同层的神经元参数添加到空列表 self.neuron_config 中
        self.neuron_config = []
        self.neuron_config.append(getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0], scaleRho=scaleRho[0]))
        self.neuron_config.append(getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1], scaleRho=scaleRho[1]))
        self.neuron_config.append(getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2], scaleRho=scaleRho[2]))

        # 取参数列表的第0个元素，即第一个神经元参数，这个配置会被传入 snn.layer(...)，这些参数会对应赋值给其内部的神经元模型
        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])

        # 是卷积层，由配置了对应参数的给自的 snn.layer 提供（slayer.py中定义了conv函数，就是调用slayer.py中conv函数，想要什么层，就在slayer.py中定义），带有脉冲特性。
        self.conv1 = self.slayer1.conv(2, 8, 5, padding=2)
        self.conv2 = self.slayer2.conv(8, 8, 3, padding=1)
        self.upconv1 = self.slayer3.convTranspose(8, 2, kernelSize=2, stride=2)

    # 这段 forward 函数是 NetworkBasic 的前向传播逻辑，用于对输入的 脉冲张量（事件数据） 进行 时空建模和上采样重建。
    def forward(self, spikeInput):
        # 通过 slayer1.psp() 对输入进行电压膜电位建模
        psp1 = self.slayer1.psp(spikeInput)

        # 输入为 [B, C, H, W, T] 形状的 5D 张量，表示一批事件数据（脉冲流）。
        # H 和 W 仍然是空间位置，代表传感器像素网格上的坐标。
        # 获取输入的维度
        B, C, H, W, T = spikeInput.shape
        # 把时间维度移到前面
        psp1_1 = psp1.permute((0, 1, 4, 2, 3))
        # 合并时间和通道，变成一堆图
        psp1_1 = psp1_1.reshape((B, C*T, H, W))
        # 空间上采样 ×2. interpolate 是 PyTorch 的插值函数，这里用 bilinear 模式对 空间维度 H 和 W 上采样 2 倍
        psp1_1 = torch.nn.functional.interpolate(psp1_1, scale_factor=2, mode='bilinear')
        # 再转回原来的维度顺序.将前面合并的 C*T 再拆开为 C 和 T，变成 [B, C, T, 2H, 2W]，
        # 然后 permute 成 [B, C, 2H, 2W, T]，与 SNN 的标准输入一致。
        psp1_1 = psp1_1.reshape(B, C, T, 2*H, 2*W).permute((0, 1, 3, 4, 2))

        # 将 PSP 结果输入脉冲卷积层，并用阈值函数转换为脉冲输出（脉冲表示事件是否激活）。
        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        # 对上一层的脉冲输出继续进行 PSP，再卷积、再脉冲。
        spikes_layer_2 = self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1)))
        # PSP 后上采样，然后与前面旁路上采样的 psp1_1 相加(像 ResNet 的残差，加入一条细节旁路路径)，再经过脉冲发放，输出最终脉冲结果。
        spikes_layer_3 = self.slayer3.spike(self.upconv1(self.slayer3.psp(spikes_layer_2)) + psp1_1)

        # 这是模型对输入低分辨率事件张量的超分重建输出。
        return spikes_layer_3


'''testing，测试了4个基于SNN的网络结构，它们的设计思路类似，但层数、卷积方式、神经元参数配置逐渐复杂'''
class Network1(torch.nn.Module):
    def __init__(self, netParams):
        super(Network1, self).__init__()
        '''
        neuron_config_conv1 = getNeuronConfig(theta=0.27, tauSr=2, tauRef=1, scaleRho=0.20)
        neuron_config_conv2 = getNeuronConfig(theta=0.25, tauSr=2, tauRef=1, scaleRho=0.13)
        neuron_config_conv3 = getNeuronConfig(theta=0.3, tauSr=4, tauRef=4, scaleRho=0.13)
        neuron_config_conv4 = getNeuronConfig(theta=0.4, tauSr=4, tauRef=4, scaleRho=0.25)
        neuron_config_conv5 = getNeuronConfig(theta=0.4, tauSr=4, tauRef=4, scaleRho=100.)
        '''

        self.neuron_config = []

        self.neuron_config.append(getNeuronConfig(theta=30, tauSr=1, tauRef=1, scaleRef=1, tauRho=1, scaleRho=10))
        self.neuron_config.append(getNeuronConfig(theta=50, tauSr=2, tauRef=2, scaleRef=1, tauRho=1, scaleRho=10))
        self.neuron_config.append(getNeuronConfig(theta=50, tauSr=2, tauRef=2, scaleRef=1, tauRho=1, scaleRho=10))
        self.neuron_config.append(getNeuronConfig(theta=100, tauSr=4, tauRef=4, scaleRef=1, tauRho=10, scaleRho=100))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])
        self.slayer4 = snn.layer(self.neuron_config[3], netParams['simulation'])

        self.conv1 = self.slayer1.conv(2, 8, 5, padding=2)
        self.conv2 = self.slayer2.conv(8, 8, 3, padding=1)
        self.conv3 = self.slayer3.conv(8, 8, 3, padding=1)
        self.upconv1 = self.slayer4.convTranspose(8, 2, 2, stride=2)

    def forward(self, spikeInput):
        psp1 = self.slayer1.psp(spikeInput)

        B, C, H, W, T = spikeInput.shape
        psp1_1 = psp1.permute((0, 1, 4, 2, 3))
        psp1_1 = psp1_1.reshape((B, C*T, H, W))
        psp1_1 = torch.nn.functional.interpolate(psp1_1, scale_factor=2, mode='bilinear')
        psp1_1 = psp1_1.reshape(B, C, T, 2*H, 2*W).permute((0, 1, 3, 4, 2))

        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        spikes_layer_2 = self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1)))
        spikes_layer_3 = self.slayer3.spike(self.conv3(self.slayer3.psp(spikes_layer_2)))
        spikes_layer_4 = self.slayer4.spike(self.upconv1(self.slayer4.psp(spikes_layer_3)) + psp1_1)

        return spikes_layer_4


class Network2(torch.nn.Module):
    def __init__(self, netParams):
        super(Network2, self).__init__()

        self.neuron_config = []
        self.neuron_config.append(getNeuronConfig(theta=30, tauSr=1, tauRef=1, scaleRef=1, tauRho=1, scaleRho=0.15))
        self.neuron_config.append(getNeuronConfig(theta=50, tauSr=2, tauRef=2, scaleRef=1, tauRho=1, scaleRho=1.5))
        self.neuron_config.append(getNeuronConfig(theta=50, tauSr=2, tauRef=2, scaleRef=1, tauRho=1, scaleRho=10))
        self.neuron_config.append(getNeuronConfig(theta=100, tauSr=4, tauRef=4, scaleRef=1, tauRho=1, scaleRho=10))
        self.neuron_config.append(getNeuronConfig(theta=400, tauSr=8, tauRef=8, scaleRef=1, tauRho=10, scaleRho=100))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])
        self.slayer4 = snn.layer(self.neuron_config[3], netParams['simulation'])
        self.slayer5 = snn.layer(self.neuron_config[4], netParams['simulation'])

        self.conv1 = self.slayer1.conv(2, 16, 5, padding=2)
        self.conv2 = self.slayer2.conv(16, 8, 1)
        self.conv3 = self.slayer3.conv(8, 8, 3, padding=1)
        self.conv4 = self.slayer4.conv(8, 16, 1)
        self.upconv1 = self.slayer5.convTranspose(16, 2, 2, stride=2)

    def forward(self, spikeInput):
        psp1 = self.slayer1.psp(spikeInput)

        B, C, H, W, T = spikeInput.shape
        psp1_1 = psp1.permute((0, 1, 4, 2, 3))
        psp1_1 = psp1_1.reshape((B, C*T, H, W))
        psp1_1 = torch.nn.functional.interpolate(psp1_1, scale_factor=2, mode='bilinear')
        psp1_1 = psp1_1.reshape(B, C, T, 2*H, 2*W).permute((0, 1, 3, 4, 2))

        conv_psp1 = self.conv1(psp1)
        spikes_layer_1 = self.slayer1.spike(conv_psp1)

        spikes_layer_2 = self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1)))
        spikes_layer_3 = self.slayer3.spike(self.conv3(self.slayer3.psp(spikes_layer_2)))
        spikes_layer_4 = self.slayer4.spike(self.conv4(self.slayer4.psp(spikes_layer_3)))

        psp5 = self.slayer5.psp(spikes_layer_4)
        conv_psp5 = self.upconv1(psp5)
        spikes_output = self.slayer5.spike(conv_psp5+psp1_1)

        return spikes_output


class Network3(torch.nn.Module):
    def __init__(self, netParams):
        super(Network3, self).__init__()

        self.neuron_config = []
        self.neuron_config.append(getNeuronConfig(theta=30, tauSr=1, tauRef=1, scaleRef=1, tauRho=1, scaleRho=0.15))
        self.neuron_config.append(getNeuronConfig(theta=50, tauSr=2, tauRef=2, scaleRef=1, tauRho=1, scaleRho=1.5))
        self.neuron_config.append(getNeuronConfig(theta=50, tauSr=2, tauRef=2, scaleRef=1, tauRho=1, scaleRho=10))
        self.neuron_config.append(getNeuronConfig(theta=100, tauSr=4, tauRef=4, scaleRef=1, tauRho=1, scaleRho=10))
        self.neuron_config.append(getNeuronConfig(theta=100, tauSr=4, tauRef=4, scaleRef=1, tauRho=1, scaleRho=10))
        self.neuron_config.append(getNeuronConfig(theta=400, tauSr=8, tauRef=8, scaleRef=1, tauRho=10, scaleRho=100))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])
        self.slayer4 = snn.layer(self.neuron_config[3], netParams['simulation'])
        self.slayer5 = snn.layer(self.neuron_config[4], netParams['simulation'])
        self.slayer6 = snn.layer(self.neuron_config[5], netParams['simulation'])

        self.conv1 = self.slayer1.conv(2, 16, 5, padding=2)
        self.conv2 = self.slayer2.conv(16, 8, 1)
        self.conv3 = self.slayer3.conv(8, 8, 3, padding=1)
        self.conv4 = self.slayer4.conv(8, 8, 3, padding=1)
        self.conv5 = self.slayer5.conv(8, 16, 1)
        self.upconv1 = self.slayer6.convTranspose(16, 2, 2, stride=2)

    def forward(self, spikeInput):
        psp1 = self.slayer1.psp(spikeInput)

        B, C, H, W, T = spikeInput.shape
        psp1_1 = psp1.permute((0, 1, 4, 2, 3))
        psp1_1 = psp1_1.reshape((B, C*T, H, W))
        psp1_1 = torch.nn.functional.interpolate(psp1_1, scale_factor=2, mode='bilinear')
        psp1_1 = psp1_1.reshape(B, C, T, 2*H, 2*W).permute((0, 1, 3, 4, 2))

        conv_psp1 = self.conv1(psp1)
        spikes_layer_1 = self.slayer1.spike(conv_psp1)

        spikes_layer_2 = self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1)))
        spikes_layer_3 = self.slayer3.spike(self.conv3(self.slayer3.psp(spikes_layer_2)))
        spikes_layer_4 = self.slayer4.spike(self.conv4(self.slayer4.psp(spikes_layer_3)))
        spikes_layer_5 = self.slayer5.spike(self.conv5(self.slayer5.psp(spikes_layer_4)))

        psp6 = self.slayer6.psp(spikes_layer_5)
        conv_psp6 = self.upconv1(psp6)
        spikes_output = self.slayer6.spike(conv_psp6+psp1_1)

        return spikes_output

if __name__ == '__main__':
    import os
    from slayerSNN.spikeFileIO import event

    def readNpSpikes(filename, timeUnit=1e-3):
        npEvent = np.load(filename)
        return event(npEvent[:, 1], npEvent[:, 2], npEvent[:, 3], npEvent[:, 0] * timeUnit * 1e3)

#
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    x = readNpSpikes('/repository/lisiqi/DVS/Classification/N-MNIST/SR_Test/LR/4/10.npy')
    x = x.toSpikeTensor(torch.zeros((2, 17, 17, 350)))
    x = torch.unsqueeze(x, dim=0).cuda()

    netParams = snn.params('./nMnist/network.yaml')
    m = NetworkBasic(netParams)
    m = torch.nn.DataParallel(m).cuda()
    with torch.no_grad():
        out = m(x)
    print((out == 0).sum(), (out == 1).sum(), ((out != 0) & (out != 1)).sum())

    # 如果是后者，这可能是正常的，是指输出了幅值为2 3 4的脉冲，而不都是单位脉冲。我记得slayersnn库输出的spike好像是有幅值的。
    # 同样，我们输入的spike有些也有幅值，由于我们将原始事件流沿时间维度堆叠到tSample个channel（e.g., tSample=350 for nMNIST dataset），
    # 在压缩过程中，如短时间内同一像素点触发多个event，会堆叠成一个倍数幅值的spike。