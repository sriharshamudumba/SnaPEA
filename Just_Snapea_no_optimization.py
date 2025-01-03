import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import matplotlib.pyplot as plt
import os

# Optimized SnaPEA Layer Wrapper
class SnaPEALayer(nn.Module):
    def __init__(self, layer, threshold=0.1):
        super(SnaPEALayer, self).__init__()
        self.layer = layer
        self.threshold = threshold

    def forward(self, x):
        # Apply threshold-based sparsity logic
        prediction_mask = (x.abs() > self.threshold).float()
        pruned_count = (prediction_mask == 0).sum().item()  # Count pruned activations
        total_count = x.numel()

        # Log pruning statistics
        print(f"Layer: {self.layer}, Pruned: {pruned_count}, Total: {total_count}, Fraction Pruned: {pruned_count/total_count:.4f}")

        # Early exit if all activations are pruned
        if prediction_mask.sum() == 0:
            print("Early exit: All activations pruned.")
            return torch.zeros_like(x)

        # Pass only significant activations
        sparse_input = x * prediction_mask
        return self.layer(sparse_input)

# SnaPEA Model Wrapper
class SnaPEAModel(nn.Module):
    def __init__(self, base_model):
        super(SnaPEAModel, self).__init__()
        self.base_model = base_model
        self._wrap_with_snapea(self.base_model)

    def _wrap_with_snapea(self, model):
        for name, module in model.named_children():
            if isinstance(module, (nn.Conv2d, nn.ReLU)):
                setattr(model, name, SnaPEALayer(module))
            else:
                self._wrap_with_snapea(module)

    def forward(self, x):
        return self.base_model(x)

# Training and Evaluation Functions
def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
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

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

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

# Visualization Function
def visualize_pruning(pruned_fractions, layer_names, output_folder="snapea_results"):
    os.makedirs(output_folder, exist_ok=True)

    # Plot pruned fractions
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(pruned_fractions)), pruned_fractions, color='orange')
    plt.xlabel("Layer Index")
    plt.ylabel("Fraction of Non-Contributing Activations")
    plt.title("Pruned Activations per Layer")
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    # Save and display plot
    plot_path = os.path.join(output_folder, "pruned_activations.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Pruned activations plot saved to {plot_path}")

# Main Function
def main():
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

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GoogLeNet with SnaPEA
    base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    snapea_model = SnaPEAModel(base_model).to(device)

    # Train and evaluate SnaPEA model
    print("Training GoogLeNet with SnaPEA...")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(snapea_model.parameters(), lr=0.001)
    train_model(snapea_model, train_loader, criterion, optimizer, device)
    snapea_accuracy, snapea_time = evaluate_model(snapea_model, test_loader, device, "SnaPEA")

    # Pruned activations visualization
    pruned_fractions = [0.1, 0.2, 0.3]  # Example fractions (replace with actual)
    layer_names = [name for name, _ in base_model.named_modules()]
    visualize_pruning(pruned_fractions, layer_names)

if __name__ == "__main__":
    main()
