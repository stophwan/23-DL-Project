import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from model import MyModel
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset")
parser.add_argument("-m", "--model", type=str, required=True, help="model parameter")
parser.add_argument("-o", "--output", type=str, required=True, help="path to output file")
args = parser.parse_args()
test_dir = args.dataset
model_path = args.model
output_dir = args.output


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path

    def _get_image_paths(self):
        image_paths = []
        image_files = sorted(os.listdir(self.root_dir))
        for image_file in image_files:
            image_path = os.path.join(self.root_dir, image_file)
            image_paths.append(image_path)
        return image_paths


transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert images to tensors
])


test_dataset = TestDataset(test_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


input_size = 256 * 256 * 3  # Adjust input size based on the size of your resized images
num_classes = 3  # Adjust based on the number of classes in your classification task

model = MyModel(input_size, num_classes)


model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

predictions = []
image_names = []
with torch.no_grad():
    for images_with_path in test_dataloader:
        images = images_with_path[0].to(device)
        image_paths = images_with_path[1]
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy().tolist())
        image_names.extend(image_paths)

class_names = ['0', '1', '2']  # Update with the class names in your task
with open(output_dir+"/output.txt", 'w') as f:
    for name, prediction in zip(image_names, predictions):
        f_name = name.split("/")[-1]
        f.write(f"{f_name}\t{prediction}\n")