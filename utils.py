import torch
import matplotlib.pyplot as plt

def compute_sparsity_loss(model):
    total = 0
    for gates in model.get_all_gates():
        total += torch.mean(gates)
    return total

def compute_sparsity(model, threshold=0.1):
    total_elements = 0
    zero_elements = 0

    for gates in model.get_all_gates():
        total_elements += gates.numel()
        zero_elements += torch.sum(gates < threshold).item()

    sparsity = 100.0 * zero_elements / total_elements
    return sparsity

def plot_gate_distribution(model, save_path="gate_distribution.png"):
    all_gates = []

    for gates in model.get_all_gates():
        all_gates.append(gates.detach().cpu().view(-1))

    all_gates = torch.cat(all_gates).numpy()

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()