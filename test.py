import torch

ce_loss = torch.nn.CrossEntropyLoss()

input = torch.randn(5, 5, 5, requires_grad=True)
target = torch.empty(5, dtype=torch.long).random_(3)
output = ce_loss(input, target)

print('input: ', input)
print('target: ', target)
print('output: ', output)