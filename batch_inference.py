
from pathlib import Path
import argparse
from src.utils.Inference import Inference
import csv
from src.data.Data import Data
from src.model.ExReNet import ExReNet
from src.utils.Config import Config
from src.utils.Inference import Inference

parser = argparse.ArgumentParser(description="")
parser.add_argument('config')
parser.add_argument('model_dir')
parser.add_argument('dataset_root')
parser.add_argument('--scale', action="store_true")
parser.add_argument('--uncertainty', action="store_true")
parser.add_argument('--legacy', action="store_true")
args = parser.parse_args()

datasets = [("chess/train", "chess/test"), ("fire/train", "fire/test"), ("heads/train", "heads/test"), ("office/train", "office/test"), ("pumpkin/train", "pumpkin/test"), ("redkitchen/train", "redkitchen/test"), ("stairs/train", "stairs/test")]
dataset_root = Path(args.dataset_root)

config = Config.from_file(args.config)
data = Data(config.get_with_prefix("data"))
model = ExReNet(config.get_with_prefix("model"), data)
model.load_weights(str(Path(args.model_dir) / "model.h5"))

mean_t, mean_R = 0, 0
for dataset in datasets:
    success, error_t, error_R = Inference(dataset_root / dataset[0], dataset_root / dataset[1], model, scale=args.scale, uncertainty=args.uncertainty, legacy_pose_transform=args.legacy).run()

    print(dataset[1] + ": " + ("%.3f" % error_t) + "m " + ("%.2f" % error_R) + "°")
    mean_t += error_t
    mean_R += error_R

mean_t /= 7
mean_R /= 7
print("Mean: " + ("%.3f" % mean_t) + "m " + ("%.2f" % mean_R) + "°")

