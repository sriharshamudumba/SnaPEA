import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

class SnaPEALayer(nn.Module):
    def __init__(self, layer, threshold=0.1):
        super(SnaPEALayer, self).__init__()
        self.layer = layer
        self.threshold = threshold

    def forward(self, x):
        prediction_mask = (x.abs() > self.threshold).float()
        sparse_input = x * prediction_mask
        return self.layer(sparse_input)

class SnaPEAModel(nn.Module):
    def __init__(self, base_model):
        super(SnaPEAModel, self).__init__()
        self.base_model = base_model
        self._wrap_with_snapea(self.base_model)

    def _wrap_with_snapea(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, SnaPEALayer(module))
            else:
                self._wrap_with_snapea(module)

    def forward(self, x):
        return self.base_model(x)

def train_and_evaluate_snapea_resnet():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    snapea_model = SnaPEAModel(base_model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(snapea_model.parameters(), lr=0.001)

    print("Training ResNet with SnaPEA...")
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = snapea_model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    print("Evaluating ResNet with SnaPEA...")
    snapea_model.eval()
    correct, total = 0, 0
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = snapea_model(inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    inference_time = time.time() - start_time
    accuracy = correct / total
    print(f"SnaPEA ResNet - Accuracy: {accuracy:.4f}, Inference Time: {inference_time:.2f}s")

if __name__ == "__main__":
    train_and_evaluate_snapea_resnet()
