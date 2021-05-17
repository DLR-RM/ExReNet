import random

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="")
parser.add_argument('pairs_file')
args = parser.parse_args()

pairs_file = Path(args.pairs_file)

with open(str(pairs_file), "r") as o:
    pairs = o.readlines()
    random.shuffle(pairs)

with open(pairs_file.with_name(pairs_file.name[:-4] + "_shuffle.txt"), "w") as o:
    o.writelines(pairs)