# PVGA.py
# probability vector-based genetic algorithms as described in "Population-Based Incremental Learning" by Shumeet Baluja
# https://www.ri.cmu.edu/pub_files/pub1/baluja_shumeet_1994_2/baluja_shumeet_1994_2.pdf

import numpy as np

class pvga:

	# fitness_function is the function that we're trying to maximize on, gene_bits is the number of bits in a genotype
	# rng is a numpy random number generator, eta is the adjustment amount, num_samples_per_gen is the number of genotypes to generate in a single generation
	# initial dist is an array of len gene_bits with initial probabilities of each bit being 1
	def __init__(self, fitness_function, gene_bits, rng, eta=0.01, num_samples_per_gen=1000, initial_dist=np.array([])):
		self.f = fitness_function
		self.gene_bits = gene_bits
		self.rng = rng
		self.num_samples = num_samples_per_gen
		self.distribution = initial_dist
		self.eta = eta
		self.median_fitness = 0
		self.max_fitness = 0
		self.highscore = 0
		# if we don't have a correct distribution, make one with 50% chance to choose each bit as 1
		if len(self.distribution) != gene_bits:
			self.distribution = np.array([0.5] * gene_bits)
		np.set_printoptions(threshold='nan')


	def create_next_generation(self, mutation_rate=0.1):
		generation = np.zeros([self.num_samples,self.gene_bits], dtype=np.uint8)
		# generate samples
		for i in range(self.gene_bits):
			generation[:,i] = np.uint8(self.rng.choice([1,0], size=self.num_samples, p=[self.distribution[i],1-self.distribution[i]]))
		fitness = self.f(generation)
		# record the median fitness for training evaluation
		self.median_fitness = np.median(fitness)
		best_sample = np.argmax(fitness)
		self.max_fitness = fitness[best_sample]
		if self.max_fitness > self.highscore: self.highscore = self.max_fitness
		self.distribution += generation[best_sample] * self.eta * fitness[best_sample]/self.highscore
		# figure out mutations
		mutate = self.rng.choice([1,-1,0], size=self.gene_bits, p=[mutation_rate/2, mutation_rate/2, 1-mutation_rate])
		self.distribution += mutate * self.eta * 10
		# elementwise max and min to bound probabilities
		self.distribution = np.maximum(self.distribution, np.zeros(self.distribution.shape))
		self.distribution = np.minimum(self.distribution, np.ones(self.distribution.shape))

	def sample_pop(self, nsamples):
		sample = np.zeros([nsamples, self.gene_bits], dtype=np.uint8)
		for i in range(self.gene_bits):
			sample[:,i] = np.uint8(self.rng.choice([1,0], size=nsamples, p=[self.distribution[i],1-self.distribution[i]]))
		return sample
 