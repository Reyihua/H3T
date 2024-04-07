import torch
a=torch.tensor([
                  [1, 5, 5, 2]
              ])
b=torch.tensor([
                  [1, 5, 5, 2]
              ])
c=a.sum()
d=b.sum()
print(c+d)
print((c+d).size())