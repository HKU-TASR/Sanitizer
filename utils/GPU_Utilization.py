import torch
import matplotlib.pyplot as plt

# 用于存储每个 epoch 的 GPU 内存使用情况
gpu_memory_allocated = []
gpu_memory_reserved = []

num_epochs = 10

for epoch in range(num_epochs):
    # 模拟训练过程
    # 你的训练代码应该在这里

    # 获取当前 GPU 内存使用情况
    allocated = torch.cuda.memory_allocated(device=1) / 1024 ** 2  # 转换为 MB
    reserved = torch.cuda.memory_reserved(device=1) / 1024 ** 2  # 转换为 MB

    # 记录 GPU 内存使用情况
    gpu_memory_allocated.append(allocated)
    gpu_memory_reserved.append(reserved)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Allocated GPU memory: {allocated:.2f} MB, Reserved GPU memory: {reserved:.2f} MB")

# 绘制 GPU 内存使用量的曲线图
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), gpu_memory_allocated, label='Allocated GPU Memory (MB)')
plt.plot(range(1, num_epochs + 1), gpu_memory_reserved, label='Reserved GPU Memory (MB)')
plt.xlabel('Epoch')
plt.ylabel('Memory (MB)')
plt.title('GPU Memory Usage Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
