import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

# Optimized SnaPEA Layer with Adaptive Sparsity
class OptimizedSnaPEALayer(nn.Module):
    def __init__(self, layer, base_percentile=75):
        super(OptimizedSnaPEALayer, self).__init__()
        self.layer = layer
        self.base_percentile = base_percentile

    def forward(self, x):
        if isinstance(self.layer, nn.Conv2d):
            # Calculate adaptive threshold per layer
            threshold = torch.quantile(x.abs(), self.base_percentile / 100.0)
            significant_mask = (x.abs() > threshold)

            # Apply sparse computation for significant activations
            sparse_input = x * significant_mask
            return self.layer(sparse_input)

        # Pass-through for non-Conv2d layers
        return self.layer(x)

# SnaPEA Model Wrapper with Improved Sparse Computation
class OptimizedSnaPEAModel(nn.Module):
    def __init__(self, base_model, base_percentile=75):
        super(OptimizedSnaPEAModel, self).__init__()
        self.base_model = base_model
        self.base_percentile = base_percentile
        self._wrap_with_snapea(self.base_model)

    def _wrap_with_snapea(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, OptimizedSnaPEALayer(module, base_percentile=self.base_percentile))
            elif isinstance(module, nn.Sequential):
                self._wrap_with_snapea(module)

    def forward(self, x):
        return self.base_model(x)

# Training Function
def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation Function
def evaluate_model(model, test_loader, device, description="Optimized SnaPEA"):
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

    accuracy = correct / total
    inference_time = time.time() - start_time
    print(f"{description} - Accuracy: {accuracy:.4f}, Inference Time: {inference_time:.2f}s")
    return accuracy, inference_time

# Main Function
def main():
    # Define number of epochs
    epochs = 5

    # Load CIFAR-10 dataset with enhanced data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet model and wrap with Optimized SnaPEA
    base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  # Use ResNet34 for better performance
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)  # Adjust final layer for CIFAR-10
    snapea_model = OptimizedSnaPEAModel(base_model, base_percentile=75).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    optimizer = AdamW(snapea_model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Train and evaluate SnaPEA model
    print("Training ResNet with Optimized SnaPEA...")
    train_model(snapea_model, train_loader, criterion, optimizer, scheduler, device, epochs=epochs)

    print("Evaluating ResNet with Optimized SnaPEA...")
    evaluate_model(snapea_model, test_loader, device, "Optimized SnaPEA")

if __name__ == "__main__":
    main()
