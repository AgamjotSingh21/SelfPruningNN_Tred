# Self Pruning Neural Network

This project implements a self-pruning neural network where weights are dynamically pruned during training using learnable gate parameters.

## Concept

Each weight has an associated gate (via sigmoid). During training:
- Gates close (→ 0) for unimportant weights  
- Important weights remain active  

Loss Function:  
Total Loss = CrossEntropyLoss + λ * L1(Gates)

## Features

- Custom PrunableLinear layer  
- Dynamic pruning during training  
- Sparsity control using L1 regularization  
- CIFAR-10 image classification  

## Project Structure
```
SelfPruningNN/
├── model.py               
├── prunable_layer.py     
├── train.py               
├── utils.py               
├── requirements.txt      
├── README.md             
├── gate_distribution.png  
├── data/         
```
## File Descriptions

- model.py  
  Implements the full neural network architecture using PrunableLinear layers.

- prunable_layer.py  
  Defines the custom linear layer where each weight has a learnable gate controlling its importance.

- train.py  
  Handles dataset loading, training loop, sparsity-aware loss computation, and evaluation.

- utils.py  
  Contains helper functions to compute sparsity loss, measure sparsity percentage, and plot gate distributions.

- requirements.txt  
  Lists all dependencies required to run the project.

- gate_distribution.png  
  Visualization of learned gate values showing pruning behavior.

- data/  
  Contains the CIFAR-10 dataset downloaded during execution.

## How to Run

pip install -r requirements.txt  
python train.py  

## Output

- Accuracy for different lambda values  
- Sparsity percentage  
- Gate distribution plot (saved as image)  

## Results 
| Lambda | Accuracy | Sparsity (%) |
|--------|----------|--------------|
| 0.01   | 46.77%   | 2.17%        |
| 0.1    | 46.69%   | 6.21%        |
| 1.0    | 46.48%   | 15.16%       |

## Observations

- Higher lambda → more pruning (higher sparsity)
- Accuracy remains relatively stable while sparsity increases


## Conclusion
- As lambda increases, sparsity increases significantly, demonstrating successful self-pruning behavior, while accuracy remains relatively stable.