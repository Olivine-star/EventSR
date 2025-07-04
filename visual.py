
import numpy as np
import matplotlib.pyplot as plt


# 加载 .npy 文件1
file_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\LR\0\0.npy"

# file_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\ResConv\HRPre\0\0.npy"
events = np.load(file_path)  # shape: (N, 4)

print(events)

# 拆解事件数据
timestamps = events[:, 0]
xs = events[:, 1].astype(int)
ys = events[:, 2].astype(int)
polarities = events[:, 3].astype(int)

# 确定图像大小（假设最大 x/y 是图像尺寸）
H = ys.max() + 1
W = xs.max() + 1

# 初始化灰色背景图像
event_image = np.full((H, W), 128, dtype=np.uint8)

# 绘制事件（正极为255，负极为0）
for x, y, p in zip(xs, ys, polarities):
    if p == 1:
        event_image[y, x] = 255
    else:
        event_image[y, x] = 0

# 显示图像
plt.figure(figsize=(5, 5))
plt.imshow(event_image, cmap='gray')
plt.title("Event Frame from .npy")
plt.axis('off')
plt.show()
