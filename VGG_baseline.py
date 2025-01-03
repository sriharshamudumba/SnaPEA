import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

# Function to train and evaluate VGG
def train_and_evaluate_vgg():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Modify VGG for CIFAR-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # Modify final layer for 10 classes
    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training VGG...")
    model.train()
    for epoch in range(5):  # Adjust epochs as needed
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate the model
    print("Evaluating VGG...")
    model.eval()
    correct, total = 0, 0
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    inference_time = time.time() - start_time
    accuracy = correct / total
    print(f"VGG - Accuracy: {accuracy:.4f}, Inference Time: {inference_time:.2f}s")
    return accuracy, inference_time

if __name__ == "__main__":
    train_and_evaluate_vgg()
