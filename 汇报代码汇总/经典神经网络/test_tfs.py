# tensorboard的使用
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
img_path = "cats_and_dogs_v2/test/cat.1452.jpg"
img_PIL = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trains = transforms.ToTensor()
img_tensor = tensor_trains(img_PIL)
print(img_tensor.shape)
writer.add_image("Tensor_image", img_tensor)
writer.close()