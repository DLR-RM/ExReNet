import argparse
from pathlib import Path

from src.data.Data import Data
from src.model.ExReNet import ExReNet
from src.utils.Config import Config
from src.utils.Inference import Inference
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2

parser = argparse.ArgumentParser(description="")
parser.add_argument('config')
parser.add_argument('model_dir')
parser.add_argument('image1')
parser.add_argument('image2')
args = parser.parse_args()

config = Config.from_file(args.config)
data = Data(config.get_with_prefix("data"))
model = ExReNet(config.get_with_prefix("model"), data)
model.load_weights(str(Path(args.model_dir) / "model.h5"))

image1 = imageio.imread(args.image1)
image1 = cv2.resize(image1, (data.image_size, data.image_size))
image2 = imageio.imread(args.image2)
image2 = cv2.resize(image2, (data.image_size, data.image_size))

cam_pose, matched_coordinates, all_dots, matching = model(image1[None] / 255.0, image2[None] / 255.0, training=False)

print("Click on the left image to see the matched point in the other image.")

full_matching = np.zeros((32, 32, 2))
for x in range(32):
    for y in range(32):
        full_matching[y, x] = matched_coordinates[-1][0, y * 4, x * 4]

fig = plt.figure()
plt.imshow(np.concatenate((image1, image2), axis=1), extent=[0, data.image_size * 2, data.image_size, 0])

for x in range(0, data.image_size * 2, 4):
    plt.plot([x, x], [0, data.image_size], 'k-', linewidth=1, alpha=0.2)

for y in range(0, data.image_size, 4):
    plt.plot([0, data.image_size * 2], [y, y], 'k-', linewidth=1, alpha=0.2)

# for (x,y) in [[13, 20]]:
#    plt.plot([x * 4 + 2, 128+ matching[y, x, 0] + 2], [y * 4 + 2, matching[y, x, 1] + 2], 'g-')

lines = []

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
    x = int(event.xdata)
    y = int(event.ydata)
    if 0 <= x < 128 and 0 <= y < 128:
        for line in lines:
            line.remove()
        lines.clear()
        x //= 4
        y //= 4
        lines.extend(plt.plot([x * 4 + 2, 128 + full_matching[y, x, 0] + 2], [y * 4 + 2, full_matching[y, x, 1] + 2], 'k-', linewidth=4))
        lines.extend(plt.plot([x * 4 + 2, 128 + full_matching[y, x, 0] + 2], [y * 4 + 2, full_matching[y, x, 1] + 2], linewidth=2))
        fig.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.axis('off')
plt.show()
# plt.savefig("match_9.png", dpi=300)
