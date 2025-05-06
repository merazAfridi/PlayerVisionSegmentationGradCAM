import torch

if torch.cuda.is_available():
    print("GPU is available")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available")

import psutil
import torch

def get_system_info():
    print("System Information")
    print("------------------")
    
    # CPU Info
    print(f"Processor: {psutil.cpu_count(logical=True)} cores")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # RAM Info
    virtual_memory = psutil.virtual_memory()
    print(f"Total RAM: {virtual_memory.total / (1024 ** 3):.2f} GB")
    print(f"Available RAM: {virtual_memory.available / (1024 ** 3):.2f} GB")
    print(f"Used RAM: {virtual_memory.used / (1024 ** 3):.2f} GB")
    
    # Disk Info
    disk_usage = psutil.disk_usage('/')
    print(f"Total Disk Space: {disk_usage.total / (1024 ** 3):.2f} GB")
    print(f"Used Disk Space: {disk_usage.used / (1024 ** 3):.2f} GB")
    print(f"Free Disk Space: {disk_usage.free / (1024 ** 3):.2f} GB")
    
    # GPU Info
    if torch.cuda.is_available():
        print("GPU is available")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Free: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
    else:
        print("GPU is not available")

if __name__ == "__main__":
    get_system_info()
