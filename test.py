import torch
import os
from PIL import Image
import torch.nn as nn
from torchvision import transforms

# Define the model structure (same as the one used during training)
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
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 28 * 28, 512),  # Ensure this matches your final feature map size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the model and the trained weights
num_classes = len(os.listdir('fruitveg/train'))  # Make sure this path is correct
model = CNNModel(num_classes=num_classes)

# Load the saved model weights
state_dict = torch.load('food_recognition_model.pth', weights_only=True)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the image transformations (should match those used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    
])

# Load and preprocess the image
image_path = 'apple.jpg'  # Replace with the path to your image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Move the image to the same device as the model
image = image.to(device)

# Perform the prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

# Map the predicted index to the corresponding class name
class_names = os.listdir('fruitveg/train')  # Make sure this path is correct
predicted_class_name = class_names[predicted.item()]

# Print the prediction
print(f'Predicted Class Index: {predicted.item()}')
print(f'Predicted Class Name: {predicted_class_name}')