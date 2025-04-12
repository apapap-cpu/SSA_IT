import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import subprocess
import re
from collections import deque
import sys
import pandas as pd

# 参数设置
NUM_LEO = 32  # LEO卫星数量
NUM_MEO = 2   # MEO卫星数量
EPOCHS = 400  # 仿真轮数
GAMMA = 0.96  # 折扣因子
ALPHA = 0.005  # 学习率
EPSILON = 0.9  # 初始epsilon值 (用于epsilon-greedy策略)
EPSILON_DECAY = 0.99  # epsilon的衰减率
MIN_EPSILON = 0.1  # 最小epsilon值
BATCH_SIZE = 32  # 批量大小
MEMORY_SIZE = 10000  # 经验回放池大小

# 任务处理参数
ALPHA_WEIGHT = 0.8  # 计算时延权重
BETA_WEIGHT = 0.8  # 传输时延权重
GAMMA_WEIGHT = 0.5  # MEO指令时延权重
TASK_SIZE = 500  # 每个任务的计算需求（MIPS）
TRANSMIT_SIZE = 100  # 传输数据大小（MB）
RESOURCE_PENALTY = -10  # 资源超限惩罚
TIME_PENALTY = -5  # 时延超限惩罚
MAX_STEPS_PER_EPOCH = 20  # 每个 epoch 至少存 20 个样本

# 加载 pkl 文件
pkl_file = '32LEO-2MEO/data.pkl'
df = pd.read_pickle(pkl_file)



def get_test_results(plan_choice):
    """执行指定的 planX.py 方案，并解析 CPU、内存使用和延迟"""
    try:
        print(f"Running: {plan_choice}")  # 打印被选择的方案
        result = subprocess.run(['python3', plan_choice], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout

        # 解析 CPU、RAM 和 Latency
        cpu_match = re.search(r'CPU MIPS: (\d+\.\d+)', output)
        ram_match = re.search(r'RAM Capacity: (\d+\.\d+)', output)
        latency_match = re.search(r'Latency: (\d+\.\d+)', output)

        if cpu_match and ram_match and latency_match:
            cpu_mips = float(cpu_match.group(1))
            ram_capacity = float(ram_match.group(1))
            latency = float(latency_match.group(1))
            return cpu_mips, ram_capacity, latency
        else:
            raise ValueError("Failed to parse output correctly")

    except Exception as e:
        print("Error during parsing:", e)
        return None, None, None


# 仿真环境
class SatelliteTaskEnv:
    def __init__(self, num_LEO, num_MEO):
        self.num_LEO = num_LEO
        self.num_MEO = num_MEO
        self.LEO_resources = []
        self.LEO_state = []
        self.groundstation_overpass = np.random.uniform(50, 300, self.num_LEO)  # 随机时间窗口
        self.transmit_time = 0
        self.dispatch_time = np.random.uniform(1, 5)  # 固定 dispatch_time
        self.epoch = 1
        self.selected_plan = None  # 记录每个 epoch 选中的 planX.py
        self.selected_MEO = None  # 记录选中的 MEO
        self.selected_LEOs = None  # 记录选中的 LEOs
        self.step_count = 0
        self.prev_total_time = 0 # 前一轮运行时延

    def get_new_state(self):
        """使用本 epoch 选定的 planX.py 获取新状态"""
        cpu, memory, latency = get_test_results(self.selected_plan)
        if cpu is None or memory is None or latency is None:
            if np.random.rand() < 0.8:
                cpu = round(np.random.uniform(4000, 10000), 2)  # 80%的概率在5000到10000之间
            else:
                cpu = round(np.random.uniform(10000, 30000), 2)  # 20%的概率在10000到30000之间     
            memory = round(np.random.uniform(4, 18), 2)
            if np.random.rand() < 0.8:
                latency = round(np.random.uniform(25, 150), 2)  # 假设时延范围 25~150
            else:
                latency = round(np.random.uniform(150,300), 2)

        return cpu, memory, latency

    def find_delivery_point(self, selected_LEOs):
        """
        在已选中的LEO卫星中找到一个交付点，使得所有选中卫星到该点的距离之和最小
        :param selected_LEO: list, 已选中的LEO卫星（可能是 0-7 或 8-15）
        :return: int, 选定的交付点卫星编号
        """
        if len(selected_LEOs) == 1:
            return selected_LEOs[0]  # 只有一个卫星，直接返回它作为交付点
    
        min_total_distance = float('inf')
        best_candidates = []
    
        for candidate in selected_LEOs:
            total_distance = sum(min(abs(candidate - other), 16 - abs(candidate - other)) for other in selected_LEOs)
        
            # 记录最小距离的候选交付点
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_candidates = [candidate]
            elif total_distance == min_total_distance:
                best_candidates.append(candidate)
    
        # 如果有多个最优候选，随机选择一个
        return random.choice(best_candidates)
    
    def calculate_transmit_time(self, selected_LEO, delivery_point, TRANSMIT_SIZE=100):
        """
        计算给定LEO卫星到交付点（delivery point）的传输时间，并返回最终 transmit_time 结果
        """

        # 如果只有 1 颗卫星，直接计算 delivery point 到地面的时间
        if len(selected_LEO) == 1:
            return TRANSMIT_SIZE / 1000  # 直接使用地面带宽 1000MHz 计算

        transmit_times = []  # 存储不同 LEO 传输到 delivery point 的时间

        for leo in selected_LEO:
            if leo == delivery_point:
                continue  # 跳过交付点本身

            # 计算 LEO 到 delivery point 的距离
            distance = min(abs(leo - delivery_point), 16 - abs(leo - delivery_point)) * 2876  # 计算实际距离 (km)

            # 根据距离选择带宽
            if distance < 5000:
                bandwidth = random.uniform(9000, 10000)  # 9000MHz-10000MHz
            elif distance < 10000:
                bandwidth = random.uniform(8000, 9000)  # 8000MHz-9000MHz
            elif distance < 15000:
                bandwidth = random.uniform(7000, 8000)  # 7000MHz-8000MHz
            elif distance < 20000:
                bandwidth = random.uniform(6000, 7000)  # 6000MHz-7000MHz
            else:
                bandwidth = random.uniform(5000, 6000)  # 5000MHz-6000MHz

            # 计算 LEO 到 delivery point 的传输时间
            t_leo = TRANSMIT_SIZE / bandwidth
            transmit_times.append(t_leo)

        # 计算 delivery point 到地面的时间
        t_ground = TRANSMIT_SIZE / 1000  # 固定 1000MHz 带宽

        # 取最大传输时间 + delivery point 到地面的时间
        transmit_time = max(transmit_times) + t_ground if transmit_times else t_ground
        return transmit_time


    def reset(self):
        """每个 epoch 重新选择一个 planX.py，并重新初始化资源"""

        if np.random.rand() < 0.8:
            cpu = round(np.random.uniform(4000, 10000), 2)  # 80%的概率在5000到10000之间
        else:
            cpu = round(np.random.uniform(10000, 30000), 2)  # 20%的概率在10000到30000之间
        
        memory = round(np.random.uniform(4, 18), 2)
        if np.random.rand() < 0.8:
            latency = round(np.random.uniform(25, 150), 2)  # 假设时延范围 25~150
        else:
            latency = round(np.random.uniform(150,300), 2)
        self.prev_state = (cpu, memory, latency)
        
        offset = int(self.prev_total_time / 5)
    
        # 读取当前 epoch 计算偏移后的对应pkl文件中的行数据
        current_row_index = max(0, min(self.epoch + offset - 1, len(df) - 1))  # 确保索引合法
        current_row = df.iloc[current_row_index]

       # 随机选定一个 MEO（0 或 1）
        self.selected_MEO = random.randint(0, self.num_MEO - 1)
    
       # 根据选定的 MEO 编号进行分类处理
        if self.selected_MEO == 0:
        # 如果选定的是MEO编号为0，则读取 pkl 文件的第 70 到 133 列
            visible_LEOs = [i for i in range(self.num_LEO) if str(current_row[70 + i * 2]).strip().upper() == 'TRUE']
        elif self.selected_MEO == 1:
        # 如果选定的是MEO编号为1，则读取 pkl 文件的第 134 到 197 列
            visible_LEOs = [i for i in range(self.num_LEO) if str(current_row[134 + i * 2]).strip().upper() == 'TRUE']
        
        # 统计0-15轨面和15-31轨面中可见卫星的数量
        visible_LEO_0_15 = [leo for leo in visible_LEOs if leo >= 0 and leo <= 15]
        visible_LEO_15_31 = [leo for leo in visible_LEOs if leo >= 15 and leo <= 31]
    
        count_0_15 = len(visible_LEO_0_15)
        count_15_31 = len(visible_LEO_15_31)
            
        visible_LEO_choose_count = max(count_0_15, count_15_31)
        # 根据可见卫星数量选择轨面
        if count_0_15 > count_15_31:
            # 如果0-7轨面可见卫星更多，从0-7轨面中选择
            selected_track = visible_LEO_0_15
        elif count_15_31 > count_0_15:
            # 如果8-15轨面可见卫星更多，从8-15轨面中选择
            selected_track = visible_LEO_15_31
        else:
            # 如果数量相同，随便选一个轨面
            selected_track = random.choice([visible_LEO_0_15, visible_LEO_15_31])
            
        print(f"Current Row: {current_row_index}, Selected MEO: {self.selected_MEO}, Visible LEOs: {visible_LEOs}, Visible LEO choose count: {visible_LEO_choose_count}")
        plan_idx = agent.act_plan(self.prev_state, visible_LEO_choose_count)  # 传递 visible_LEO_count , 选择 0-7 对应 plan1-plan8

        self.selected_plan = [f'plan{i}.py' for i in range(1, 9)][plan_idx]
        
        cpu_new, memory_new, latency_new = self.get_new_state()  # 运行 plan 后得到的新状态
        
        # 解析 plan 选定的 LEO 数量
        num_LEO_to_use = int(self.selected_plan.replace("plan", "").replace(".py", ""))
    
        selected_LEOs = random.sample(selected_track, num_LEO_to_use)
    
        self.selected_LEOs = selected_LEOs
        delivery_point = self.find_delivery_point(self.selected_LEOs)
        self.transmit_time = self.calculate_transmit_time(self.selected_LEOs, delivery_point, TRANSMIT_SIZE)
        
        # **计算新的 total_latency**
        total_latency = latency_new + self.transmit_time
        
        # **更新 self.current_state**
        self.current_state = (cpu_new, memory_new, total_latency)  # 存储新的三元组
 
        self.LEO_resources = [self.current_state] * self.num_LEO 
        self.LEO_state = [self.prev_state] * self.num_LEO

        print(f"Initial State at Epoch {self.epoch}: {self.prev_state}")
        print(f"Selected Plan: {self.selected_plan}, MEO: {self.selected_MEO}, LEOs: {self.selected_LEOs}")
        print(f"Selected delivery point: {delivery_point}, Transmit Time: {self.transmit_time:.4f} s")
        

        # ============ 计算 self.groundstation_overpass ============
        # **获取 `delivery_point` 对应的仰角数据列**
        elevation_col = 6 + delivery_point * 2  # 直接使用 `delivery_point` 计算列索引
        elevation_data = df.iloc[:, elevation_col].values  # 读取 `delivery_point` 这一列的所有仰角数据

        # **寻找 `delivery_point` 何时仰角从正变负**
        ground_contact_time = None
        for j in range(current_row_index, len(elevation_data) - 1):  # 从当前行往后找
            if elevation_data[j] > 0 and elevation_data[j + 1] <= 0:
                # **找到仰角从正变负的点**
                ground_contact_time = (j + 1 - current_row_index) * 5
                break

        # **确保 `self.groundstation_overpass` 取到了合理值**
        if ground_contact_time is None:
            raise ValueError(f"No valid ground contact time found for delivery_point {delivery_point}!")

        print(f"Groundstation Overpass (Delivery Point {delivery_point}): {ground_contact_time}")
       
        self.groundstation_overpass = np.full(self.num_LEO, ground_contact_time)
        self.dispatch_time = np.random.uniform(1, 5)
        self.step_count = 0
        self.epoch += 1
        return np.array(self.LEO_resources).flatten()

    def step(self, action):
        total_time = 0
        reward = 0
        valid_leo_count = 0  # 记录有效任务的LEO数量
        planned_leo_count = len(self.selected_LEOs)
  
        for leo in self.selected_LEOs:
            meo = self.selected_MEO
            if action[leo][meo] > 0:
                workload_ratio = action[leo][meo]
                task_cpu = (workload_ratio * TASK_SIZE)
                task_memory = (workload_ratio * TRANSMIT_SIZE) / 1024

                cpu, memory, total_latency = self.LEO_resources[leo]
                print(cpu, memory, total_latency)
                if task_cpu > cpu or task_memory > memory :
                    print(f"Skipping LEO {leo} due to resource limits! Task Memory:{task_memory}")
                    reward += RESOURCE_PENALTY
                    compute_time = self.current_state[2] - self.transmit_time
                    continue

                compute_time = self.current_state[2] - self.transmit_time
                valid_leo_count += 1

                if total_latency  > self.groundstation_overpass[leo]:
                    reward += TIME_PENALTY
    
            else:
                print(f'action[{leo}][{meo}] = {action[leo][meo]}, allocation failed!')
    

        reward -= (ALPHA_WEIGHT * compute_time + BETA_WEIGHT * self.transmit_time + GAMMA_WEIGHT * self.dispatch_time)

        self.step_count += 1
        done = (self.step_count >= MAX_STEPS_PER_EPOCH)
        total_time = total_latency + self.dispatch_time
        if valid_leo_count == planned_leo_count:
            print(f"Final time calculation: {compute_time}, {self.transmit_time}, {self.dispatch_time}")
        else:
            total_time = float('inf')
            print(f"Plan Failed! Valid LEO Count: {valid_leo_count}, Planned LEO Count: {planned_leo_count}")

        return np.array(self.LEO_state).flatten(), reward, done, total_time, action



# DQN 代理 改进使用Double_DQN
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.plan_action_size = 8  # 8 种 plan 选择
        self.task_action_size = action_size  # 任务分配
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA

        # **创建两个 Q 网络（plan_model & target_model）**
        self.plan_model = self._build_model(self.plan_action_size)  # 在线 Q 网络
        self.target_model = self._build_model(self.plan_action_size)  # 目标 Q 网络
        self.target_model.load_state_dict(self.plan_model.state_dict())  # 初始化为相同参数

        # 任务分配网络
        self.task_model = self._build_model(self.task_action_size)

        self.optimizer = optim.Adam(self.plan_model.parameters(), lr=ALPHA)
        self.criterion = nn.MSELoss()

    def _build_model(self, action_size):
        """构建 Q 网络"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),  # 第一个全连接层
            nn.ReLU(),  # 激活函数
            nn.Linear(128, 64),  # 第二个全连接层
            nn.ReLU(),  # 激活函数
            nn.Linear(64, action_size),  # 输出层，大小为 action_size
            # 去掉 Sigmoid 激活函数，直接线性输出
        )
    
        # 使用 Xavier 初始化方法初始化权重
        for m in model:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # Xavier 初始化方法
                if m.bias is not None:
                    init.zeros_(m.bias)  # 偏置初始化为零
    
        return model


    def act_plan(self, state, visible_LEO_choose_count):
        """Double DQN 选择最优 `planX.py`"""
        state = np.array(state)
        if state.shape[0] == 3:  # 如果 `state` 是 `(1, 3)`，扩展为 `(1, 96)`
            state = np.tile(state, 32)  # 把单个 LEO 状态扩展到 32 个 LEO

        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)  # 确保形状是 `(1, 96)`

        if np.random.rand() <= self.epsilon :
            return np.random.choice(min(visible_LEO_choose_count, 8))  # 4 种 plan 选择 (0-3)

        action_values = self.plan_model(state_tensor)  # 选择 `plan_model` 网络
        # 获取最大可行的 plan 数量
        max_plan_count = min(visible_LEO_choose_count, 8)  # 最大只能选择可见 LEO 数量以内的 plan

        # 获取符合条件的最大 Q 值的计划
        valid_plans = []
        if action_values.dim() == 2:  # action_values 是二维张量 (1, 3)
            for i in range(max_plan_count):  # 遍历 0 到 max_plan_count-1 的范围
                valid_plans.append(action_values[0, i].item())  # 使用 action_values[0, i] 获取第一个 batch 的第 i 个值
        else:
            print("Error: unexpected action_values dimensions")

        print(f"valid_plans: {valid_plans}")  # 打印 valid_plans，帮助调试
        # 获取 Q 值最大的计划
        plan_idx = valid_plans.index(max(valid_plans))  # 获取 Q 值最大的 plan
        return plan_idx  # 返回最优计划

    def act_task(self, state):
        """DQN 控制任务分配"""
        action = np.zeros((NUM_LEO, NUM_MEO))
        if np.random.rand() <= self.epsilon:
            for leo in env.selected_LEOs:
                action[leo][env.selected_MEO] = np.random.rand()
        else:
            state_tensor = torch.FloatTensor(state)
            act_values = self.task_model(state_tensor).detach().numpy().reshape(NUM_LEO, NUM_MEO)
            act_values += 1e-6  # 防止归一化时出现 0
            act_values /= act_values.sum()  # 归一化任务分配
            act_values = np.clip(act_values, 0, 1)
            for leo in env.selected_LEOs:
                action[leo][env.selected_MEO] = act_values[leo][env.selected_MEO]
                action[leo][env.selected_MEO] = max(action[leo][env.selected_MEO], 1e-6)
        return action

    def remember(self, state, plan_action, task_action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, plan_action, task_action, reward, next_state, done))

    def replay(self, batch_size):
        """Double DQN 训练"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, plan_action, task_action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # **Double DQN 目标 Q 计算**
            with torch.no_grad():
                # 1. **用 `plan_model` 选择下一步最优 action**
                next_action = torch.argmax(self.plan_model(next_state_tensor)).item()

                # 2. **用 `target_model` 计算 Q 值**
                next_q_value = self.target_model(next_state_tensor)[next_action].detach()

                # 3. 计算目标 Q 值
                target = reward + (self.gamma * next_q_value if not done else 0)

            # 计算当前状态 Q 值
            predicted_q_values = self.plan_model(state_tensor)
            predicted_q_value = predicted_q_values[plan_action]

            # 计算损失
            loss = self.criterion(predicted_q_value, torch.tensor(target, dtype = torch.float32))

            # 反向传播优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 逐步更新 ε（epsilon-greedy 探索策略）
        if len(self.memory) < 6000:  # 早期阶段
            self.epsilon = max(0.5, self.epsilon * self.epsilon_decay)  # 不能低于 0.5
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # 进入后期稳定探索

    def evaluate_policy_improvement(self):
        """评估 Double DQN 选取的 action 是否接近最优"""

        sample_size = min(500, len(self.memory))  # 采样 500 个状态进行评估
        print(len(self.memory))
        minibatch = random.sample(self.memory, sample_size)

        q_differences = []

        for state, plan_action, _, _, _, _ in minibatch:
            state_tensor = torch.FloatTensor(state)
            if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
                print("[Policy Improvement] Warning: state contains NaN or Inf. Skipping this sample.")
                continue


            # **1. 计算 DQN 选取的 action 的 Q 值**
            chosen_q_value = self.plan_model(state_tensor)[plan_action].item()
            if torch.isnan(torch.tensor(chosen_q_value)) or torch.isinf(torch.tensor(chosen_q_value)):
                print("[Policy Improvement] Warning: chosen_q_value is NaN or Inf. Skipping this sample.")
                continue

            # **2. 计算该状态下所有 action 的最大 Q 值**
            max_q_value = torch.max(self.plan_model(state_tensor)).item()
            if torch.isnan(torch.tensor(max_q_value)) or torch.isinf(torch.tensor(max_q_value)):
                print("[Policy Improvement] Warning: max_q_value is NaN or Inf. Skipping this sample.")
                continue

            # **3. 计算 Q 值误差**
            q_difference = chosen_q_value - max_q_value
            q_differences.append(q_difference)
        
        if len(q_differences) == 0:
            print("[Policy Improvement] Warning: No valid Q-differences found. Skipping calculation.")
            return None

        # **计算误差均值**
        expectation_gap = np.mean(q_differences)
        print(f"[Policy Improvement Check] Expectation Gap: {expectation_gap:.4f}")
        return expectation_gap

    def update_target_model(self):
        """软更新目标网络"""
        self.target_model.load_state_dict(self.plan_model.state_dict())


# 训练
env = SatelliteTaskEnv(NUM_LEO, NUM_MEO)
agent = DQNAgent(NUM_LEO * 3, NUM_LEO * NUM_MEO)
reward_list = [] # 记录每个epoch的平均奖励


# 创建或清空 RL_output.txt（如果文件已存在，则清空）
with open("RL_output3.txt", "w") as f:
    f.write("RL Training Output Log\n\n")

# 重新定义 print()，让其同时写入文件和终端
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)  # 在终端输出
        self.log.write(message)  # 在文件中写入
        self.log.flush()  # 立即写入文件，防止断电丢失数据

    def flush(self):
        pass  # 兼容性保留，通常 `sys.stdout` 需要 `flush()`

sys.stdout = Logger("RL_output3.txt")  # 重定向 print() 到文件

if not os.path.exists("model3"):
    os.makedirs("model3")

for e in range(1, EPOCHS + 1):
    print(f"EPOCH {e}")
    state = env.reset()
    done = False
    best_strategy = None
    min_total_time = float("inf")
    total_reward = 0 # 记录累计奖励
    step_count = 0
    while not done:
        action = agent.act_task(state)
        next_state, reward, done, total_time, current_strategy = env.step(action)
        plan_idx = ['plan1.py', 'plan2.py', 'plan3.py', 'plan4.py', 'plan5.py', 'plan6.py', 'plan7.py', 'plan8.py'].index(env.selected_plan)
        
        agent.remember(state, plan_idx, action, reward, next_state, done)
        agent.replay(BATCH_SIZE)
        
        total_reward += (GAMMA ** step_count) * reward
        step_count += 1
        state = next_state

        # 记录最优策略
        if total_time < min_total_time:
            total_latency = total_time
            best_strategy = current_strategy
            min_total_time = total_time
    
    agent.update_target_model()
    # 输出每个 epoch 的最佳策略和最小时延
    env.prev_total_time += total_latency
    
    reward_list.append(total_reward)
    average_reward = np.mean(reward_list)  # 计算每个 epoch 的平均奖励
    
    expectation_gap = agent.evaluate_policy_improvement()
    if expectation_gap is None:
        print(f"Epoch {e} : Expectation Gap not calculated (memory too small).")
    else:
        print(f"Epoch {e} : Expectation Gap = {expectation_gap:.4f}")

    print(f"Epoch {e} : Average Reward = {average_reward:.4f}")
    print(f"Best Strategy = {best_strategy}, Total Latency = {total_latency:.4f} s")
    if e % 20 == 0:  # 每 20 个 epoch 保存一次模型
        # ======== 保存模型 ========
        model_filename = f"model3/dqn_epoch_{e}.pth"
        torch.save({
            'epoch': e,
            'plan_model_state_dict': agent.plan_model.state_dict(),
            'task_model_state_dict': agent.task_model.state_dict(),
            'target_model_state_dict': agent.target_model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        }, model_filename)
        print(f"Model saved at {model_filename}")