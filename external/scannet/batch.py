from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('dataset_path')
args = parser.parse_args()

for path in Path(args.dataset_path).iterdir():
    print(path)
    ret = os.system('cd external/scannet && python3 reader.py  --filename ' + str(path / (path.name + '.sens')) + ' --output_path ' + str(path.parent / path.name) + ' --export_color_images --export_depth_images --export_poses --export_intrinsics --width 640 --height 480 --frame_skip 10')
    if ret != 0:
        exit(0)