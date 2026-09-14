#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os
import sys

ID = 3

#Platform Name: Apple
#Platform Vendor: Apple
#    Device Name: Apple M3
#    Device Type: ALL | GPU
#    Device Vendor: Apple
#    Device Version: OpenCL 1.2

# 0 : MacBook Pro (13-inch, 2017, Two Thunderbolt 3 ports)
#     macOS Monterey Version 12.3.1
#     opencl
# 1 : Threadripper
#     ubuntu
#     opencl
# 2 : Nvidia DGX
#     ubuntu
#     cupy
# 3 : macOS Metal

#if sys.platform.startswith('darwin'):
#    import opencl
#else:
if ID==0 or ID==1:
    import opencl
elif ID==2:
    import dgx
elif ID==3:
    import lmetal
#

def getGpu(idx=0):
    if ID==0: # MBP
        platform_id = 0
        device_id = 0
        # 0 : Intel(R) Core(TM) i7-7660U CPU @ 2.50GH
        # 1 : Intel(R) Iris(TM) Plus Graphics 64
        # 2 : AMD Radeon Pro 580 Compute Engine
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif ID==1: # tr
        platform_id = 1
        device_id = 0
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif ID==2: # nvidia
        my_gpu = dgx.Dgx(idx)
    elif ID==3: # macOS Metal
        my_gpu = lmetal.LMetal()
        my_gpu.prepare()
    else:
        print("error : undefined platform")
        my_gpu = None
    #
    return my_gpu

