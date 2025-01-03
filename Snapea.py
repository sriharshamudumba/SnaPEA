import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

# Optimized SnaPEA Layer with Sparse Computation
class OptimizedSnaPEALayer(nn.Module):
    def __init__(self, layer, percentile=95):
        super(OptimizedSnaPEALayer, self).__init__()
        self.layer = layer
        self.percentile = percentile

    def forward(self, x):
        if isinstance(self.layer, nn.Conv2d):
            # Calculate dynamic threshold
            threshold = torch.quantile(x.abs(), self.percentile / 100.0)
            significant_mask = (x.abs() > threshold)

            # Apply sparse computation for significant activations
            sparse_input = x * significant_mask
            if sparse_input.sum() == 0:  # Early exit for all-zero input
                return torch.zeros_like(x)

            return self.layer(sparse_input)

        # Pass-through for non-Conv2d layers
        return self.layer(x)

# SnaPEA Model Wrapper with Reduced Input Size
class OptimizedSnaPEAModel(nn.Module):
    def __init__(self, base_model):
        super(OptimizedSnaPEAModel, self).__init__()
        self.base_model = base_model
        self._wrap_with_snapea(self.base_model)

    def _wrap_with_snapea(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(model, name, OptimizedSnaPEALayer(module))
            elif isinstance(module, nn.Sequential):
                self._wrap_with_snapea(module)

    def forward(self, x):
        return self.base_model(x)

# Training Function with AMP
def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation Function
def evaluate_model(model, test_loader, device, description="SnaPEA"):
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
    # Load CIFAR-10 dataset with reduced input size
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Reduce input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increased batch size
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GoogLeNet model and wrap with SnaPEA
    base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)  # Adjust final layer for CIFAR-10
    snapea_model = OptimizedSnaPEAModel(base_model).to(device)

    # Train and evaluate SnaPEA model
    print("Training GoogLeNet with Optimized SnaPEA...")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(snapea_model.parameters(), lr=0.001)
    train_model(snapea_model, train_loader, criterion, optimizer, device)

    print("Evaluating GoogLeNet with Optimized SnaPEA...")
    evaluate_model(snapea_model, test_loader, device, "Optimized SnaPEA")

if __name__ == "__main__":
    main()
