"""
输出格式示例 - 演示双通道模型的不同输出格式

这个脚本展示了双通道 EventSR 模型如何输出不同格式：
1. 事件流格式 (Event Stream) - [B, C, H, W, T]
2. 事件帧格式 (Event Frame) - [B, C, H, W]
"""

import torch
import sys
sys.path.append('.')

from dual_channel_model import create_dual_channel_model


def demonstrate_output_formats():
    """演示不同的输出格式"""
    
    print("="*60)
    print("双通道 EventSR 输出格式演示")
    print("="*60)
    
    # 创建示例输入
    batch_size = 2
    channels = 2  # 正负极性
    height, width = 17, 17
    time_steps = 350
    
    # 输入：事件流格式 [B, C, H, W, T]
    input_spikes = torch.randn(batch_size, channels, height, width, time_steps)
    input_spikes = (input_spikes > 0).float()  # 转换为二进制脉冲
    
    print(f"输入事件流格式: {input_spikes.shape}")
    print(f"输入事件总数: {input_spikes.sum().item()}")
    
    # 基础配置
    base_config = {
        'netParams': {'simulation': {'Ts': 1, 'tSample': 350}},
        'snn_model_type': 'NetworkBasic',
        'cnn_base_channels': 32,  # 减小以便快速演示
        'fusion_strategy': 'adaptive',
        'event_frame_strategy': 'time_based',
        'num_event_frames': 8,
        'scale_factor': 2,
        'use_temporal_cnn': False
    }
    
    print("\n" + "-"*50)
    print("1. 事件流输出格式 (Event Stream)")
    print("-"*50)
    
    # 配置1：事件流输出
    config_stream = base_config.copy()
    config_stream['output_format'] = 'event_stream'
    
    model_stream = create_dual_channel_model(config_stream)
    model_stream.eval()
    
    with torch.no_grad():
        outputs_stream = model_stream(input_spikes)
    
    print("输出结果:")
    print(f"  最终输出 (output): {outputs_stream['output'].shape}")
    print(f"  SNN 通道输出: {outputs_stream['snn_output'].shape}")
    print(f"  CNN 通道输出: {outputs_stream['cnn_output'].shape}")
    print(f"  融合特征: {outputs_stream['fused_features'].shape}")
    
    # 分析事件流输出
    final_output_stream = outputs_stream['output']
    print(f"\n事件流输出分析:")
    print(f"  形状: {final_output_stream.shape} - [批次, 通道, 高度, 宽度, 时间]")
    print(f"  时间维度: {final_output_stream.shape[-1]} 时间步")
    print(f"  空间分辨率: {final_output_stream.shape[2]}x{final_output_stream.shape[3]}")
    print(f"  输出事件总数: {final_output_stream.sum().item():.1f}")
    print(f"  每时间步平均事件数: {final_output_stream.sum().item() / final_output_stream.shape[-1]:.2f}")
    
    print("\n" + "-"*50)
    print("2. 事件帧输出格式 (Event Frame)")
    print("-"*50)
    
    # 配置2：事件帧输出
    config_frame = base_config.copy()
    config_frame['output_format'] = 'event_frame'
    
    model_frame = create_dual_channel_model(config_frame)
    model_frame.eval()
    
    with torch.no_grad():
        outputs_frame = model_frame(input_spikes)
    
    print("输出结果:")
    print(f"  最终输出 (output): {outputs_frame['output'].shape}")
    print(f"  SNN 通道输出: {outputs_frame['snn_output'].shape}")
    print(f"  CNN 通道输出: {outputs_frame['cnn_output'].shape}")
    print(f"  融合特征: {outputs_frame['fused_features'].shape}")
    
    # 分析事件帧输出
    final_output_frame = outputs_frame['output']
    print(f"\n事件帧输出分析:")
    print(f"  形状: {final_output_frame.shape} - [批次, 通道, 高度, 宽度]")
    print(f"  无时间维度 - 静态图像格式")
    print(f"  空间分辨率: {final_output_frame.shape[2]}x{final_output_frame.shape[3]}")
    print(f"  像素值范围: [{final_output_frame.min().item():.3f}, {final_output_frame.max().item():.3f}]")
    print(f"  像素值总和: {final_output_frame.sum().item():.1f}")
    
    print("\n" + "-"*50)
    print("3. 输出格式对比")
    print("-"*50)
    
    print("事件流格式 (Event Stream):")
    print("  ✓ 保留完整的时序信息")
    print("  ✓ 与原始 EventSR 兼容")
    print("  ✓ 适合需要时序分析的应用")
    print("  ✓ 可以重建事件序列")
    print("  - 内存占用较大")
    
    print("\n事件帧格式 (Event Frame):")
    print("  ✓ 内存占用较小")
    print("  ✓ 适合传统图像处理")
    print("  ✓ 便于可视化和分析")
    print("  ✓ 与标准CNN输出兼容")
    print("  - 丢失时序信息")
    
    print("\n" + "-"*50)
    print("4. 各通道输出分析")
    print("-"*50)
    
    snn_output = outputs_stream['snn_output']
    cnn_output = outputs_stream['cnn_output']
    
    print(f"SNN 通道 (保留时序):")
    print(f"  输出格式: {snn_output.shape} - 事件流")
    print(f"  特点: 时序建模，脉冲输出")
    
    print(f"\nCNN 通道 (空间特征):")
    print(f"  输出格式: {cnn_output.shape} - 事件帧")
    print(f"  特点: 空间特征提取，连续值输出")
    
    print(f"\n融合结果:")
    print(f"  结合了SNN的时序信息和CNN的空间特征")
    print(f"  输出格式可配置：事件流或事件帧")
    
    return outputs_stream, outputs_frame


def compare_memory_usage():
    """比较不同输出格式的内存使用"""
    
    print("\n" + "="*60)
    print("内存使用对比")
    print("="*60)
    
    # 示例张量大小
    batch_size = 4
    channels = 2
    height, width = 34, 34  # 超分辨率后的尺寸
    time_steps = 350
    
    # 事件流格式内存
    stream_elements = batch_size * channels * height * width * time_steps
    stream_memory_mb = stream_elements * 4 / (1024 * 1024)  # float32 = 4 bytes
    
    # 事件帧格式内存
    frame_elements = batch_size * channels * height * width
    frame_memory_mb = frame_elements * 4 / (1024 * 1024)
    
    print(f"批次大小: {batch_size}")
    print(f"输出分辨率: {height}x{width}")
    print(f"时间步数: {time_steps}")
    
    print(f"\n事件流格式:")
    print(f"  张量大小: {stream_elements:,} 元素")
    print(f"  内存占用: {stream_memory_mb:.2f} MB")
    
    print(f"\n事件帧格式:")
    print(f"  张量大小: {frame_elements:,} 元素")
    print(f"  内存占用: {frame_memory_mb:.2f} MB")
    
    print(f"\n内存节省:")
    print(f"  节省比例: {(1 - frame_memory_mb/stream_memory_mb)*100:.1f}%")
    print(f"  节省内存: {stream_memory_mb - frame_memory_mb:.2f} MB")


def usage_recommendations():
    """使用建议"""
    
    print("\n" + "="*60)
    print("使用建议")
    print("="*60)
    
    print("选择事件流格式 (event_stream) 当:")
    print("  • 需要分析时序动态")
    print("  • 与原始EventSR模型对比")
    print("  • 进行时序一致性评估")
    print("  • 需要重建完整事件序列")
    
    print("\n选择事件帧格式 (event_frame) 当:")
    print("  • 内存资源有限")
    print("  • 只关注空间超分辨率质量")
    print("  • 与传统图像处理方法对比")
    print("  • 进行可视化展示")
    
    print("\n配置方法:")
    print("  在配置文件中设置:")
    print("    output_format: 'event_stream'  # 或 'event_frame'")
    print("  或在代码中设置:")
    print("    config['output_format'] = 'event_frame'")


def main():
    """主函数"""
    try:
        # 演示输出格式
        outputs_stream, outputs_frame = demonstrate_output_formats()
        
        # 内存使用对比
        compare_memory_usage()
        
        # 使用建议
        usage_recommendations()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("请检查依赖和安装")


if __name__ == '__main__':
    main()
