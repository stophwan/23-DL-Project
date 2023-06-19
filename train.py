import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import argparse
from PIL import Image
import os
from model import MyModel
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset")
parser.add_argument("-m", "--model", type=str, required=True, help="path to model parameter")
args = parser.parse_args()
root_dir = args.dataset
save_path = args.model

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._get_image_paths_and_labels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def _get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        class_dirs = sorted(os.listdir(self.root_dir))
        for class_index, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                image_files = sorted(os.listdir(class_path))
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    image_paths.append(image_path)
                    labels.append(class_index)
        return image_paths, labels

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert images to tensors
])

dataset = ImageDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


input_size = 256 * 256 * 3  # Adjust input size based on the size of your resized images
num_classes = 3

model = MyModel(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy*100:.2f}%")

print("Training finished!")

torch.save(model.state_dict(), save_path)
print("Model parameters saved successfully!")