from pathlib import Path
import tensorflow as tf

from src.data.Data import Data
from src.model.ExReNet import ExReNet
from src.train.Trainer import Trainer
from src.utils.Config import Config
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description="")
parser.add_argument('config')
parser.add_argument('task_dir')
parser.add_argument('num_iterations', type=int)
parser.add_argument('--save_interval', type=int, default=100)
args = parser.parse_args()

task_dir = Path(args.task_dir)
task_dir.mkdir(exist_ok=True, parents=True)

config = Config.from_file(args.config)
data = Data(config.get_with_prefix("data"))
model = ExReNet(config.get_with_prefix("model"), data)
trainer = Trainer(config.get_with_prefix("train"), model, data)

weights_path = task_dir / Path("model.h5")
if weights_path.exists():
    print("Loading weights from " + str(weights_path))
    model.load_weights(str(weights_path))

tensorboard_writer = tf.summary.create_file_writer(str(task_dir))

for current_iteration in tqdm(range(args.num_iterations)):
    trainer.step(tensorboard_writer, current_iteration)
    if current_iteration > 0 and current_iteration % args.save_interval == 0:
        print("Auto save after " + str(current_iteration) + " iterations")
        model.save_weights(str(weights_path))

model.save_weights(str(weights_path))

