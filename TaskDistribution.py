from SineDistribution import *

class TaskDistribution(object):
	def __init__(self, min_amplitude=0.5, max_amplitude=5):
		self.min_amplitude = min_amplitude
		self.max_amplitude = max_amplitude

	def sample_task(self):
		amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
		phase = 0 if random.choice([True, False]) else math.pi
		return SineDistribution(amplitude, phase)
