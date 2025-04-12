import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import time
import os
import numpy as np

def parse_tegrastats_log(logfile, script_name, max_time):
    data = {
        'time': [],
        'cpu_usage': [],
        'ram_usage': [],
        'script': []
    }
    
    with open(logfile, 'r') as f:
        for line in f:
            match = re.search(r'RAM (\d+)/(\d+)MB .*? CPU \[([0-9%@,]+)\]', line)
            if match:
                ram_used, ram_total, cpu_usage_str = match.groups()
                ram_usage = (int(ram_used) / int(ram_total)) * 100
                cpu_usages = list(map(lambda x: int(x.split('%')[0]), cpu_usage_str.split(',')))
                cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
                data['time'].append(len(data['time']))  # 记录采样点的索引
                data['cpu_usage'].append(cpu_usage)
                data['ram_usage'].append(ram_usage)
                data['script'].append(script_name)
    
    df = pd.DataFrame(data)
    
    # 归一化时间轴到 max_time
    if not df.empty:
        df['time'] = np.linspace(0, max_time, num=len(df['time']))
    
    return df

def plot_resource_usage(df):
    # 设置字体大小（默认字体大小的5倍，假设默认大小为12）
    font_size = 63  # 标题和坐标轴标签
    legend_font_size = 40  # 图例
    tick_font_size = 30  # 刻度标签
    
    # 调整图像尺寸和布局
    fig, axs = plt.subplots(2, 1, figsize=(35, 21), sharex=True)  # 原为 (12, 6)
    
    scripts = df['script'].unique()
    colors = ['b', 'g', 'r', 'c']  # 颜色列表
    
    for i, script in enumerate(scripts):
        script_df = df[df['script'] == script]
        axs[0].plot(script_df['time'].to_numpy(), script_df['ram_usage'].to_numpy(), 
                    label=script_labels.get(script, script), color=colors[i % len(colors)], linewidth=3)
        axs[1].plot(script_df['time'].to_numpy(), script_df['cpu_usage'].to_numpy(), 
                    label=script_labels.get(script, script), color=colors[i % len(colors)], linewidth=3)
    
    # 设置RAM子图
    axs[0].set_ylabel('RAM Usage (%)', fontsize=font_size)
    axs[0].set_ylim(10, 100)
    axs[0].legend(fontsize=legend_font_size, loc='upper right')
    axs[0].tick_params(axis='both', labelsize=tick_font_size)
    
    # 设置CPU子图
    axs[1].set_ylabel('CPU Usage (%)', fontsize=font_size)
    axs[1].set_ylim(0, 100)
    axs[1].legend(fontsize=legend_font_size, loc='upper right')
    axs[1].tick_params(axis='both', labelsize=tick_font_size)
    
    # 设置共用X轴标签
    plt.xlabel('Time (seconds)', fontsize=font_size)
    plt.tight_layout(pad=5.0)  # 增加子图间距，防止重叠
    plt.savefig("resource_usage_comparison.png", dpi=100, bbox_inches='tight')

def start_tegrastats():
    return subprocess.Popen(["tegrastats", "--logfile", "tegrastats.log"])

def monitor_script(script_name, max_time):
    # 清空日志
    open('tegrastats.log', 'w').close()
    
    # 启动监控
    tegrastats_process = start_tegrastats()
    time.sleep(20)  # 运行前20秒数据收集
    
    # 运行目标脚本
    subprocess.run(["python3", script_name])
    
    # 运行后继续收集数据
    time.sleep(40)
    
    # 终止监控
    tegrastats_process.terminate()
    tegrastats_process.wait()
    
    # 解析日志
    return parse_tegrastats_log('tegrastats.log', script_name, max_time)

if __name__ == '__main__':
    scripts = ["it_start.py", "overlap.py", "overlap3.py", "o2.py"]
    script_labels = {"it_start.py": "plan1", "overlap.py": "plan2", "overlap3.py": "plan4", "o2.py": "plan8"}
    all_data = pd.DataFrame()
    max_time = 300  # 统一时间轴长度
    
    for script in scripts:
        df = monitor_script(script, max_time)
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    # 绘制结果
    plot_resource_usage(all_data)