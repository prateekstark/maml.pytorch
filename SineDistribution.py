import random
import math
import torch

class SineDistribution(object):
	def __init__(self, amplitude=1, phase=0):
		self.amplitude = amplitude
		self.phase = phase

	def sample_data(self, batch_size=1):
		X = (-10) * torch.rand(batch_size, 1) + 5
		y = self.amplitude * torch.sin(X - self.phase)
		return X, y

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import numpy as np
	sineCurve = SineDistribution(amplitude=1, phase=0)
	X, y = sineCurve.sample_data(10000)
	# Plotting Sine Curve
	X_sorted, indices = torch.sort(X, dim=0)
	plt.plot(X_sorted, y[indices][:, :, 0])
	plt.show()
