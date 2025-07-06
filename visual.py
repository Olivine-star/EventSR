# # 绘制事件（正极为255，负极为0），事件表示方法是该像素最后时刻发生的极性是什么就是什么
# import numpy as np
# import matplotlib.pyplot as plt

# # -------- 设置显示完整数组 --------
# np.set_printoptions(threshold=np.inf)

# # 加载 .npy 文件
# file_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\LR\0\0.npy"
# # file_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\ResConv\HRPre\0\0.npy"

# events = np.load(file_path)  # shape: (N, 4)

# # 完整打印所有事件，245到305ms间发生的事件
# print("事件数据如下：")
# print(events)


# # 拆解事件数据
# timestamps = events[:, 0]
# xs = events[:, 1].astype(int)
# ys = events[:, 2].astype(int)
# polarities = events[:, 3].astype(int)

# # 确定图像大小（假设最大 x/y 是图像尺寸）
# H = ys.max() + 1
# W = xs.max() + 1

# # 初始化灰色背景图像
# event_image = np.full((H, W), 128, dtype=np.uint8)


# # 绘制事件（正极为255，负极为0），事件表示方法是该像素最后时刻发生的极性是什么就是什么
# for x, y, p in zip(xs, ys, polarities):
#     if p == 1:
#         event_image[y, x] = 255
#     else:
#         event_image[y, x] = 0

# # 显示图像
# plt.figure(figsize=(5, 5))
# plt.imshow(event_image, cmap='gray')
# plt.title("Event Frame from .npy")
# plt.axis('off')
# plt.show()



# # 绘制事件（正极为255，负极为0），事件表示方法是该像素的事件累计
import numpy as np
import matplotlib.pyplot as plt

# 读取事件数据
file_path = r"D:\PycharmProjects\EventSR-dataset\dataset\N-MNIST\SR_Train\LR\0\0.npy"
events = np.load(file_path)  # shape: (N, 4)

# 拆解
xs = events[:, 1].astype(int)
ys = events[:, 2].astype(int)

# 获取图像大小
H = ys.max() + 1
W = xs.max() + 1

# 初始化累计图像
event_count = np.zeros((H, W), dtype=np.uint16)  # 统计事件次数

# 事件频率累计（不分正负）
for x, y in zip(xs, ys):
    event_count[y, x] += 1

# 可选：归一化到 0-255 用于显示
event_image = event_count.astype(np.float32)
event_image = 255 * (event_image / event_image.max())  # 归一化
event_image = event_image.astype(np.uint8)

# 显示累计事件热图
plt.figure(figsize=(5, 5))
plt.imshow(event_image, cmap='hot')  # 可以试试 'hot', 'gray', 'viridis'
plt.title("Event Frequency Map (Accumulated Count)")
plt.axis('off')
plt.colorbar(label='Event Count')
plt.show()
