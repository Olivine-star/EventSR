import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

# 读取事件流路径123
# npy_path = r'D:\PycharmProjects\EventSR\SR-ES1\EventStream-SR-main\dataset\N-MNIST\SR_Train\LR\0\0.npy'
npy_path = r'D:\PycharmProjects\EventSR\SR-ES1\EventStream-SR-main\dataset\N-MNIST\ResConv\HRPre\0\0.npy'
gif_output_path = r'D:\PycharmProjects\EventSR-dataset\result_visual\0_HRPre.gif'

# 加载事件流 (N, 4)
events = np.load(npy_path)
print("事件流 shape:", events.shape)

# 参数设定
H, W = 34, 34  # N-MNIST 的分辨率
num_bins = 100  # 将整个时间切成多少段
x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

# 归一化时间戳，并映射到 bin
t = t - t.min()
t = t / t.max()
t_bins = (t * (num_bins - 1)).astype(int)

# 初始化事件帧
event_frames = np.zeros((num_bins, H, W), dtype=np.float32)

# 累积事件到帧中
for i in range(events.shape[0]):
    xi, yi, bi = int(x[i]), int(y[i]), t_bins[i]
    pol = 1 if p[i] > 0 else -1
    if 0 <= yi < H and 0 <= xi < W:
        event_frames[bi, yi, xi] += pol

# 可视化生成 gif
frames = []
for i in range(num_bins):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(event_frames[i], cmap='gray', vmin=-1, vmax=1)
    ax.axis('off')
    tmp_path = f'tmp_{i:03d}.png'
    plt.savefig(tmp_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    frames.append(imageio.imread(tmp_path))
    os.remove(tmp_path)

# 保存 gif
imageio.mimsave(gif_output_path, frames, duration=0.1)
print(f"GIF 已保存至: {gif_output_path}")
