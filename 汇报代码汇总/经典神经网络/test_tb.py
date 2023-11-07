
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "cats_and_dogs_v2/train/cats/cat.0.jpg"
img_PLI = Image.open(image_path)
img_arr = np.array(img_PLI)
print(type(img_arr))
print(img_arr.shape)

writer.add_image("test", img_arr, 1, dataformats='HWC')

for i in range(0, 100):
    writer.add_scalar("y = 2x", 3 * i, i)

writer.close()
