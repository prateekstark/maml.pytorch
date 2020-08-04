import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(1, 40)
		self.fc2 = nn.Linear(40, 40)
		self.fc3 = nn.Linear(40, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
    
	def set_params(self, params):
		with torch.no_grad():
			self.fc1.weight.data = params[0]
			self.fc1.bias.data = params[1]
			self.fc2.weight.data = params[2]
			self.fc2.bias.data = params[3]
			self.fc3.weight.data = params[4]
			self.fc3.bias.data = params[5]

	def parameterized_forward(self, x, params):
		x = F.relu(F.linear(x, params[0], params[1]))
		x = F.relu(F.linear(x, params[2], params[3]))
		x = F.linear(x, params[4], params[5])
		return x