import numpy as np
from pathlib import Path
import argparse
import h5py
from tqdm import tqdm

parser = argparse.ArgumentParser(description="")
parser.add_argument('path')
parser.add_argument('key', 'densevlad')
args = parser.parse_args()

root = Path(args.path)

densevlads = []

for path in tqdm(list(sorted(root.rglob("*.hdf5"), key=lambda file: int(file.name.split(".")[0])))):
    with h5py.File(str(path), 'r') as f:
        densevlads.append(f[args.key][:])

print(len(densevlads))
densevlads = np.stack(densevlads, axis=0)

np.savez_compressed(str(root / (args.key + "_lookup")), densevlads)
