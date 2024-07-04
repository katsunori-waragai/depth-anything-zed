"""
This code is only to check available cuda device
"""
import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")
    for device in range(device_count):
        device_name = torch.cuda.get_device_name(device)
        print(f"Device {device}: {device_name}")
else:
    print("No CUDA devices available.")
