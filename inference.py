
import argparse
from pathlib import Path

from src.data.Data import Data
from src.model.ExReNet import ExReNet
from src.utils.Config import Config
from src.utils.Inference import Inference

parser = argparse.ArgumentParser(description="")
parser.add_argument('config')
parser.add_argument('model_dir')
parser.add_argument('ref_path')
parser.add_argument('query_path')
parser.add_argument('--scale', action="store_true")
parser.add_argument('--uncertainty', action="store_true")
parser.add_argument('--legacy', action="store_true")
args = parser.parse_args()

config = Config.from_file(args.config)
data = Data(config.get_with_prefix("data"))
model = ExReNet(config.get_with_prefix("model"), data)
model.load_weights(str(Path(args.model_dir) / "model.h5"))

result = Inference(args.ref_path, args.query_path, model, args.scale, args.uncertainty, args.legacy).run()
print(result)
