import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.data.Data import Data
from src.utils.Config import Config

parser = argparse.ArgumentParser(description="")
parser.add_argument('config')
args = parser.parse_args()

config = Config.from_file(args.config)
data = Data(config.get_with_prefix("data"))

dataset = data.build_val_dataset()

for reference_images, reference_cam_poses, query_images, query_cam_poses, iou, room_ids, pose_transform, full_matching in dataset:
    fig = plt.figure()
    plt.imshow(np.concatenate((reference_images[0], query_images[0]), axis=1), extent=[0, data.image_size * 2, data.image_size, 0])

    lines = []

    def onclick(event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
        x = int(event.xdata)
        y = int(event.ydata)
        if 0 <= x < 128 and 0 <= y < 128:
            for line in lines:
                line.remove()
            lines.clear()
            lines.extend(plt.plot([x, 128 + full_matching[0, y, x, 0]], [y, full_matching[0, y, x, 1]], 'k-', linewidth=4))
            lines.extend(plt.plot([x, 128 + full_matching[0, y, x, 0]], [y, full_matching[0, y, x, 1]], linewidth=2))
            fig.canvas.draw()


    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    full_matching[0].numpy().tofile("test.raw")

    plt.axis('off')
    plt.show(False)
    fig2 = plt.figure()
    plt.imshow(np.concatenate((full_matching[0] / 128, np.zeros_like(full_matching[0, ..., 0:1])), -1))
    plt.show()
