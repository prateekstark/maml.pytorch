from model import *
from TaskDistribution import *
from utils import *
from copy import deepcopy
import matplotlib.pyplot as plt
import logging

class MAML(object):
	def __init__(self, num_iterations=1000, num_task=1000, batch_size=10, inner_steps=1, alpha=0.03, beta=0.01, task_distribution=None, device=torch.device("cpu"), logger=None):
		self.device = device
		self.model = Net().to(device)
		self.params = list(self.model.parameters())
		self.num_iterations = num_iterations
		self.num_task = num_task
		self.batch_size = batch_size
		self.criterion = nn.MSELoss()
		self.inner_steps = inner_steps
		self.alpha = alpha
		self.beta = beta
		self.optimizer = torch.optim.Adam(self.params, beta)
		self.meta_loss = []
		self.task_distribution = task_distribution
		self.logger = logger

	def pretrain(self):
		total_loss = 0
		for iterations in range(self.num_iterations):
			meta_loss_sum = 0
			for _ in range(self.num_task):
				task = self.task_distribution.sample_task()
				theta_dash = [theta.clone() for theta in self.params]
				X, y = task.sample_data(self.batch_size)
				for _ in range(self.inner_steps):
					loss = self.criterion(self.model.parameterized_forward(X.to(device), theta_dash), y.to(device)) / self.batch_size
					grads = torch.autograd.grad(loss, theta_dash)
					theta_dash = [w - self.alpha * g for w, g in zip(theta_dash, grads)]
				X, y = task.sample_data(self.batch_size)
				loss = self.criterion(self.model.parameterized_forward(X.to(device), theta_dash), y.to(device)) / self.batch_size
				meta_loss_sum += loss
			meta_grads = torch.autograd.grad(meta_loss_sum, self.params, allow_unused=True)
			for w, g in zip(self.params, meta_grads):
				w.grad = g
			self.optimizer.step()
			total_loss += meta_loss_sum / self.num_task
			if(iterations%10 == 0):
				self.logger.info("Iteration: " + str(iterations))
				self.logger.info("Total Loss:" + str(total_loss))
				self.meta_loss.append(total_loss)
		self.model.set_params(self.params)

def finetune_model(model, data, criterion, num_steps, batch_size, optim = 'SGD'):
	if(optim == 'Adam'):
		optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	else:
		optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	
	X, y = data
	losses = []
	
	for step in range(num_steps):
		optimizer.zero_grad()
		output = model(X.to(device))
		loss = criterion(output, y.to(device)) / batch_size
		losses.append(loss.item())
		loss.backward()
		optimizer.step()
	return losses

if __name__ == '__main__':

	logging.basicConfig(filename='logfile.log', format='%(levelname)s %(asctime)s %(message)s', filemode='w')
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)

	if torch.cuda.is_available():
		device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
		logger.info("Running on the GPU")
	else:
		device = torch.device("cpu")
		logger.info("Running on the CPU")

	task_distribution = TaskDistribution()
	maml = MAML(num_iterations=501, task_distribution=task_distribution, device=device, logger=logger)
	logger.info("MAML model initialized...")
	random_model = deepcopy(maml.model)
	logger.info("MAML pretraining started...")
	maml.pretrain()
	pretrained_model = deepcopy(maml.model)	

	# Testing on random tasks

	batch_size = 10
	num_steps = 10
	criterion = nn.MSELoss()
	task = task_distribution.sample_task()
	data = task.sample_data(batch_size=batch_size)

	logger.info("MAML finetuning started...")
	maml_losses = finetune_model(pretrained_model, data, criterion, num_steps, batch_size)
	logger.info("Random Model finetuning started...")
	random_model_losses = finetune_model(random_model, data, criterion, num_steps, batch_size)

	logging.info("Saving plots")
	plot_loss(random_model_losses, maml_losses)
	plot_curve(random_model, pretrained_model, task, device)