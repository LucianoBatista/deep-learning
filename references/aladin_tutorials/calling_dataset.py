from sklearn.utils import shuffle
from custom_dataset import CatsAndDogsDataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


dataset = CatsAndDogsDataset(
    csv_file="cats_dogs.csv",
    root_dir="data/cats_dogs/cats_dogs_resized",
    transform=transforms.ToTensor(),
)

train_ds, test_ds = torch.utils.data.random_split(dataset, [8, 2])
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True)

for image, label in train_ds:
    np_img = image.numpy().transpose((1, 2, 0))
    plt.imshow(np_img, cmap="gray")
    plt.show()
