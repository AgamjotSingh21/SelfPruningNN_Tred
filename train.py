import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import SelfPruningNN
from utils import compute_sparsity_loss, compute_sparsity, plot_gate_distribution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False)

def train_model(lambda_val):
    model = SelfPruningNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            classification_loss = criterion(outputs, labels)
            sparsity_loss = compute_sparsity_loss(model)

            loss = classification_loss + lambda_val * sparsity_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Lambda {lambda_val} | Epoch {epoch+1} | Loss: {running_loss:.4f}")

    return model

def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    lambdas = [0.01, 0.1, 1]

    results = []

    for l in lambdas:
        print(f"\nTraining with lambda = {l}")

        model = train_model(l)

        acc = evaluate(model)
        sparsity = compute_sparsity(model)

        print(f"Lambda: {l} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

        results.append((l, acc, sparsity))

        if l == lambdas[-1]:
            plot_gate_distribution(model)

    print("\nFinal Results:")
    for r in results:
        print(f"Lambda: {r[0]}, Accuracy: {r[1]:.2f}%, Sparsity: {r[2]:.2f}%")