# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import copy

# # Set parameters
# input_size = 128
# batch_size = 32
# num_classes = 36  # Make sure this matches your dataset
# num_epochs = 50
# learning_rate = 0.001
# data_dir = 'fruitveg'

# # Data preprocessing function
# def preprocess_image(image):
#     if image.mode in ("RGBA", "P"):
#         image = image.convert("RGB")
#     return image

# # Data augmentation and preprocessing
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Lambda(preprocess_image),  # Apply preprocessing
#         transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
        
#     ]),
#     'val': transforms.Compose([
#         transforms.Lambda(preprocess_image),  # Apply preprocessing
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
       
#     ]),
# }

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['train', 'val']}
# dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
#                for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes
# num_classes = len(class_names)  # Update to match dataset

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Define the CNN model
# class CNNModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CNNModel, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(128 * (input_size // 8) * (input_size // 8), 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# model = CNNModel(num_classes=num_classes).to(device)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Training the model
# def train_model(model, criterion, optimizer, num_epochs=25):
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     train_loss_history = []
#     val_loss_history = []
#     train_acc_history = []
#     val_acc_history = []

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#             if phase == 'train':
#                 train_loss_history.append(epoch_loss)
#                 train_acc_history.append(epoch_acc)
#             else:
#                 val_loss_history.append(epoch_loss)
#                 val_acc_history.append(epoch_acc)

#     print(f'Best val Acc: {best_acc:4f}')

#     model.load_state_dict(best_model_wts)
#     return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# if __name__ == '__main__':
#     # Train and evaluate
#     model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(model, criterion, optimizer, num_epochs=num_epochs)

#     # Save the model
#     torch.save(model.state_dict(), 'food_recognition_model.pth')

#     # Plotting the training and validation accuracy and loss
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_acc_history, label='Training Accuracy')
#     plt.plot(val_acc_history, label='Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.title('Training and Validation Accuracy')

#     plt.subplot(1, 2, 2)
#     plt.plot(train_loss_history, label='Training Loss')
#     plt.plot(val_loss_history, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Training and Validation Loss')

#     plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image
import matplotlib.pyplot as plt

# Custom Dataset
class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load data
        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):  # Check if it's a directory
                for img_file in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, img_file)
                    if os.path.isfile(img_path):  # Check if it's an image file
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root_dir = 'fruitveg/train'
dataset = FoodDataset(root_dir=root_dir, transform=transform)

# Check if dataset is loaded
if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check the directory structure and files.")

# Train-test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

# Data Loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Use the flattened_size calculated earlier
        flattened_size = self._get_flattened_size()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def _get_flattened_size(self):
        # Calculate the flattened size of the output from the feature extractor
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = self.features(dummy_input)
            return output.view(-1).size(0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Calculate number of classes based on the dataset
num_classes = len(os.listdir('fruitveg/train'))
model = CNNModel(num_classes=num_classes)

# Print details
print(f'Number of classes: {num_classes}')
print(f'Unique labels in dataset: {set(dataset.labels)}')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 25

# Lists to store training and validation metrics
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# Training and evaluation
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.item())
    
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    val_loss = running_loss / len(test_loader.dataset)
    val_acc = running_corrects.double() / len(test_loader.dataset)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc.item())
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Save the model
model_path = 'food_recognition_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# Plotting
plt.figure(figsize=(12, 5))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()  # Return the class index as an integer

# Predict on a sample image
image_path = 'apple.png'

# Ensure model is on the correct device (e.g., GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load an image, apply the same transformations, and predict
predicted_class = predict_image(image_path, model, transform)
print(f'Predicted Class Index: {predicted_class}')

# Check if the predicted class index corresponds to the correct class name
class_names = os.listdir('fruitveg/train')  # Get class names
predicted_class_name = class_names[predicted_class]
print(f'Predicted Class Name: {predicted_class_name}')