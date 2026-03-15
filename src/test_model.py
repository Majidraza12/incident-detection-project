from model import CNNLSTM
import torch

model = CNNLSTM()
x = torch.randn(4, 16, 3, 224, 224)   # batch of 4 videos
y = model(x)

print(y.shape)