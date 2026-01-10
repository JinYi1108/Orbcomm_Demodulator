import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from orbdemod import orbdemod

data = np.fromfile('./raw_data/20250415-1940-0.dat', dtype = np.int16)

packets = orbdemod(
    raw_data=data[0:480_000_000],
    log_level="INFO",           
    plot=True, 
    plot_save_dir="./examples/pipeline_results", 
    output_file="./examples/pipeline_orbdemod_packets.txt"
)

print(f"解调出 {len(packets)} 个有效包")