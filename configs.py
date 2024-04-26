import argparse
import torch
import numpy as np
import pynvml
import os

def get_best_gpu():
    """Return gpu (:class:`torch.device`) with largest free memory."""
    assert torch.cuda.is_available()
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys() is not None:
        cuda_devices = [
            int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        ]
    else:
        cuda_devices = range(deviceCount)

    assert max(cuda_devices) < deviceCount
    deviceMemory = []
    for i in cuda_devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d" % (best_device_index))

def get_args():
    parser = argparse.ArgumentParser(description='manual to this script')
    # model
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=3)
    parser.add_argument('--hidden_layer', type=int, default=2048, help='ffn layer')
    
    # optimizer
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup',type=int, default=4000)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epsilon_ls', type=float, default=1e-9)
    parser.add_argument('--clip', type=float, default=1.0)

    args = parser.parse_args()
    return args 