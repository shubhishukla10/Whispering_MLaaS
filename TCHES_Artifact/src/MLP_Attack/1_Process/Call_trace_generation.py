from time import perf_counter
import numpy as np
import pandas as pd
import ctypes
import pathlib
import os
from pathlib import Path
home = str(Path.home())

base_path = home + "/TCHES_Artifact/"

libname = base_path + "utils/lib_flush.so"
flush_lib = ctypes.CDLL(libname)

libname = base_path + "utils/lib_flush_pipe.so"
flush_lib_pipe = ctypes.CDLL(libname)


for i in range(1,101):
    flush_lib.main()
    flush_lib_pipe.main()
    os.system("taskset -c 0 python Generate_timing_samples.py "+str(i))
    


