import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((120,320)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.5,), (0.5,))])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 11 * 33, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)
        self.fc4 = nn.Linear(7, 1)

    def forward(self, x):
        #print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
# # 定义求导函数
# def get_Variable(x):
#     x = torch.autograd.Variable(x)  # Pytorch 的自动求导
#     # 判断是否有可用的 GPU
#     return x.cuda() if torch.cuda.is_available() else x

# 判断是否GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()

loaded_model = torch.load('state_dict.pt', map_location=torch.device('cpu'))
model.load_state_dict(loaded_model)
model.eval()


input_names = ['input']
output_names = ['output']

x = torch.randn(1, 1, 120, 320, requires_grad=True)  # 这个要与你的训练模型网络输入一致。我的是黑白图像

torch.onnx.export(model, x, 'state_dict.onnx', input_names=input_names, output_names=output_names, verbose='True')