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
            with open(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", "r") as f:
                cpu_freq = int(f.read().strip()) / 1000  # 转换为 MHz
                cpu_freqs.append(cpu_freq)
        
        cpu_mhz = sum(cpu_freqs) / len(cpu_freqs) if cpu_freqs else 0
    except Exception as e:
        print(f"Error reading CPU frequency: {e}")

    total_mem = 0
    available_mem = 0
    try:
        mem_info = subprocess.check_output("cat /proc/meminfo", shell=True).decode("utf-8")
        for line in mem_info.splitlines():
            if "MemTotal" in line:
                total_mem = int(line.split(":")[1].strip().split()[0])  
            elif "MemAvailable" in line:
                available_mem = int(line.split(":")[1].strip().split()[0])  

        total_mem_mb = total_mem / 1024  
        available_mem_mb = available_mem / 1024  
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

        remaining_ram = total_ram - (avg_ram_usage / 100) * total_ram
        if remaining_ram >= 1024:  
            remaining_ram_str = f"{remaining_ram / 1024:.2f} GB"
        else:
            remaining_ram_str = f"{remaining_ram:.2f} MB"

        return remaining_ram_str

    except Exception as e:
        print(f"Error reading tegrastats log: {e}")
        return "N/A"

def run_and_monitor(script_name):
    logfile = 'tegrastats.log'
    if os.path.exists(logfile):
        with open(logfile, 'w') as f:
            f.truncate(0)

    cpu_freq, total_ram, remaining_mem = get_system_info()
    

    try:
        # 运行 `perf` 并监测 `script_name` 执行的指令数
        perf_command = f"/home/nvidia/.local/bin/perf stat -e instructions -x, python3 {script_name}"
        process = subprocess.Popen(perf_command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        stdout, stderr = process.communicate()
        perf_output = stderr.decode('utf-8')

        instructions = 0
        for line in perf_output.splitlines():
            if "instructions" in line:
                parts = line.split(",")
                if len(parts) > 0:
                    instructions = int(parts[0].replace(" ", "").replace(",", ""))
                    break 

        # 记录程序执行时间
        start_time = time.time()
        process = subprocess.Popen(['python3', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        end_time = time.time()

        latency = end_time - start_time

        if latency > 0 and instructions > 0:
            mips = instructions / (latency * 10**6)
            return mips, remaining_mem, latency

    except subprocess.CalledProcessError as e:
        print(f"Error running perf: {e}")
    
    return None, remaining_mem, None

def compare_and_output():
    # 运行 o1.py
    o1_mips, o1_remaining_mem, o1_latency = run_and_monitor('o1.py')

    # 运行 n3.py
    n3_cpu_mips, n3_remaining_mem, n3_latency = run_and_monitor('n3.py')

    # 计算最大值
    max_mips = max(o1_mips, n3_cpu_mips) if o1_mips and n3_cpu_mips else None
    max_remaining_mem = max(o1_remaining_mem, n3_remaining_mem, key=lambda x: float(x.split()[0]))
    max_latency = max(o1_latency, n3_latency)

    # 输出最大值
    if max_mips is not None:
        print(f"CPU MIPS: {max_mips:.2f}")
    else:
        print("MIPS calculation failed.")

    print(f"RAM Capacity: {max_remaining_mem}")
    print(f"Latency: {max_latency:.6f} s")



# 运行比较函数
compare_and_output()