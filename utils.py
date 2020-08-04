import matplotlib.pyplot as plt
import torch

def evaluate_model(model, data, criterion):
	X, y = data
	with torch.zero_grad():
		output = model(X)
		loss = criterion(output, y)
	return loss, output

def plot_loss(other_model_loss, maml_loss):
	plt.figure()
	plt.plot(other_model_loss, label='other_loss')
	plt.plot(maml_loss, label='maml_loss')
	plt.legend()
	plt.savefig('loss.png')

def plot_curve(other_model, maml_model, task, device):
	X, y = task.sample_data(10000)
	X_sorted, indices = torch.sort(X, dim=0)
	y_other = other_model(X_sorted.to(device)).cpu().detach()
	y_maml = maml_model(X_sorted.to(device)).cpu().detach()
	plt.figure()
	plt.plot(X_sorted, y[indices][:, :, 0], label='real_curve')
	plt.plot(X_sorted, y_other, label='other_model_curve')
	plt.plot(X_sorted, y_maml, label='maml_curve')
	plt.legend()
	plt.savefig('networks.png')
