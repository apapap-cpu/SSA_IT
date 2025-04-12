import subprocess
import re
import time
import os

def get_system_info():
    # 获取 CPU 主频（MHz）
    cpu_mhz = 0
    try:
        cpu_freqs = []
        for i in range(os.cpu_count()):
            # 读取每个核心的频率（单位为 kHz）
            with open(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", "r") as f:
                cpu_freq = int(f.read().strip()) / 1000  # 转换为 MHz
                cpu_freqs.append(cpu_freq)
        
        # 计算 CPU 的平均频率
        cpu_mhz = sum(cpu_freqs) / len(cpu_freqs) if cpu_freqs else 0
    except Exception as e:
        print(f"Error reading CPU frequency: {e}")

    # 获取总内存容量和剩余内存
    total_mem = 0
    available_mem = 0
    try:
        mem_info = subprocess.check_output("cat /proc/meminfo", shell=True).decode("utf-8")
        for line in mem_info.splitlines():
            if "MemTotal" in line:
                total_mem = int(line.split(":")[1].strip().split()[0])  # 单位为 kB
            elif "MemAvailable" in line:
                available_mem = int(line.split(":")[1].strip().split()[0])  # 单位为 kB

        # 将内存从 kB 转换为 MB 或 GB
        total_mem_mb = total_mem / 1024  # 单位 MB
        available_mem_mb = available_mem / 1024  # 单位 MB
        if available_mem_mb >= 1024:
            available_mem_str = f"{available_mem_mb / 1024:.2f} GB"
        else:
            available_mem_str = f"{available_mem_mb:.2f} MB"

    except Exception as e:
        print(f"Error reading memory info: {e}")

    return cpu_mhz, total_mem_mb, available_mem_str

def parse_tegrastats_log(logfile, duration, total_ram):
    data = {'ram_usage': []}
    try:
        with open(logfile, 'r') as f:
            for line in f:
                match = re.search(r'RAM (\d+)/(\d+)MB', line)
                if match:
                    ram_used, ram_total = match.groups()
                    ram_usage = (int(ram_used) / int(ram_total)) * 100
                    data['ram_usage'].append(ram_usage)

        avg_ram_usage = sum(data['ram_usage']) / len(data['ram_usage']) if data['ram_usage'] else 0

        # 计算剩余内存容量（单位 MB 或 GB）
        remaining_ram = total_ram - (avg_ram_usage / 100) * total_ram
        if remaining_ram >= 1024:  # 如果剩余内存大于等于 1024MB，显示为 GB
            remaining_ram_str = f"{remaining_ram / 1024:.2f} GB"
        else:
            remaining_ram_str = f"{remaining_ram:.2f} MB"

        return remaining_ram_str

    except Exception as e:
        print(f"Error reading tegrastats log: {e}")
        return "N/A"

def run_and_monitor():
    logfile = 'tegrastats.log'
    if os.path.exists(logfile):
        with open(logfile, 'w') as f:
            f.truncate(0)

    # 获取实际的 CPU 频率和内存信息
    cpu_freq, total_ram, remaining_mem = get_system_info()

    

    # 使用 perf 获取指令条数并计算 MIPS
    try:
        # 使用 perf 统计程序的指令条数
        perf_command = "/home/nvidia/.local/bin/perf stat -e instructions -x, python3 it_start.py"
        process = subprocess.Popen(perf_command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        # 获取 stderr（因为 perf 统计信息在 stderr 中）
        stdout, stderr = process.communicate()
        perf_output = stderr.decode('utf-8')  # perf 统计信息在 stderr 里


        # 解析 perf 输出中的指令条数
        instructions = 0
        for line in perf_output.splitlines():
            if "instructions" in line:
                parts = line.split(",")
                if len(parts) > 0:
                    instructions = int(parts[0].replace(" ", "").replace(",", ""))  # 去掉逗号和空格，转换为整数
                    break 

        # 获取程序的执行时间（latency）
        start_time = time.time()
        process = subprocess.Popen(['python3', 'it_start.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        end_time = time.time()

        latency = end_time - start_time

        # 计算 MIPS
        if latency > 0 and instructions > 0:
            mips = instructions / (latency * 10**6)
            return mips, remaining_mem, latency

    except subprocess.CalledProcessError as e:
        print(f"Error running perf: {e}")
    
    return None, remaining_mem, None

def compare_and_output():
    # 运行 it_start.py 并获取结果
    mips, remaining_mem, latency = run_and_monitor()

    if mips is not None:
        print(f"CPU MIPS: {mips:.2f}")
    else:
        print("MIPS calculation failed.")
    
    print(f"RAM Capacity: {remaining_mem}")
    print(f"Latency: {latency:.6f} s")

# 运行比较函数
compare_and_output()
