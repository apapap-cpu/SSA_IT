import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 文件路径
file_path = 'RL_output_dqn(final).txt'

# 初始化数据存储
epochs = []
expectation_gaps = []
average_rewards = []

# 使用正则表达式提取信息
with open(file_path, 'r') as file:
    for line in file:
        # 匹配包含 "Epoch" 和 "Expectation Gap"
        match_gap = re.search(r"Epoch (\d+) :.*Expectation Gap = (-?\d+\.\d+)", line)
        # 匹配包含 "Epoch" 和 "Average Reward"
        match_reward = re.search(r"Epoch (\d+) :.*Average Reward = (-?\d+\.\d+)", line)
        
        # 如果匹配到 "Expectation Gap"
        if match_gap:
            epoch = int(match_gap.group(1))
            if epoch > 15:
                gap_value = float(match_gap.group(2))
                # 存储 "Expectation Gap"
                epochs.append(epoch)
                expectation_gaps.append(gap_value)
        
        # 如果匹配到 "Average Reward"
        if match_reward:
            epoch = int(match_reward.group(1))
            if epoch > 15:
                reward_value = float(match_reward.group(2))
                # 存储 "Average Reward"
                average_rewards.append(reward_value)

# 检查数据长度，确保数据完整
print(f"Epochs: {len(epochs)}, Expectation Gaps: {len(expectation_gaps)}, Average Rewards: {len(average_rewards)}")

# 检查数据是否完整
assert len(epochs) == len(expectation_gaps) == len(average_rewards)

# 设置字体大小
font_size = 23  # 默认字体大小的5倍（假设默认大小为4）

# 绘制 Expectation Gap 单独的图
# 创建图像
fig, ax = plt.subplots(figsize=(18, 14))
ax.plot(epochs, expectation_gaps, marker='o', color='b')

# 设置坐标轴标签
font_size = 24
ax.set_xlabel('Epoch', fontsize=24)
ax.set_ylabel('Expectation Gap', fontsize=24)

# 开启网格、图例
ax.grid(True)
ax.legend(fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=24)

# 使用 ScalarFormatter 强制启用科学计数法，并控制格式
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))  # 设置何时触发科学计数法
ax.yaxis.set_major_formatter(formatter)

# 放大左上角的科学计数单位（OffsetText）
ax.yaxis.get_offset_text().set_size(font_size)  # 放大左上角单位文本

plt.show()

# 绘制 Average Reward 单独的图
plt.figure(figsize=(18, 14))
plt.plot(epochs, average_rewards, marker='o', color='r')
plt.xlabel('Epoch', fontsize=font_size)
plt.ylabel('Average Reward', fontsize=24)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=font_size)  # 调整刻度标签大小
plt.show()



