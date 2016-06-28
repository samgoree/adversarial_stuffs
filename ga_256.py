# ga_256.py
# Identical to ga.py, but uses values in between 0 and 255 for elements of a genotype

import numpy as np

class ga:

	# each element of the population is a numpy array of 1's and 0's with length equal to gene_bits
	# rng should be a numpy random number generator
	# initial_pop should be a numpy array of size [population_size, gene_bits] or less
	# fitness_function should take a vector of bitvectors and assign a fitness value to each one
	# diversity limit is the minimum standard deviation before the generation is considered undiverse and randomly reinitialized
	def __init__(self, fitness_function, gene_bits, rng, population_size=1000, diversity_limit=.01, initial_pop=np.array([], dtype=np.uint8)):
		self.f = fitness_function
		self.pop = initial_pop
		self.pop_size = population_size
		self.rng = rng
		self.gene_bits = gene_bits
		self.diversity_limit = diversity_limit
		if len(self.pop) < self.pop_size:
			self.pop=np.reshape(np.append(self.pop, self.rng.randint(low=0,high=256,size=[self.pop_size - len(self.pop), self.gene_bits])), [self.pop_size,self.gene_bits])
		self.median_fitness = np.median(self.f(self.pop))


	# applies the fitness function, then randomly chooses mutation_rate out of the remaining population and mutates them randomly
	def create_next_generation(self, mutation_rate=0.1):
		# figure out the fitness score
		fitness_scores = self.f(self.pop)
		self.median_fitness = np.median(fitness_scores)
		print "Standard deviation: ", np.std(fitness_scores)
		# if the minimum score is less than 10% above the median, the population diversity has collapsed and we restart from scratch
		if (np.std(fitness_scores)) < self.diversity_limit:
			print "median == min, population fitness collapsed, randomly generating new generation"
			self.pop = np.reshape(self.rng.randint(0,2,[self.pop_size, self.gene_bits]), [self.pop_size,self.gene_bits])
			print self.pop.shape
		# apply natural selection (I read somehwere that compress works faster than fancy indexing, so I'm using it)
		self.pop = np.compress(fitness_scores > self.median_fitness, self.pop, axis=0)
		# mutate genes
		genes_to_mutate = self.rng.choice(np.arange(len(self.pop)), len(self.pop) * mutation_rate, replace=False)
		self.pop[genes_to_mutate,:] = self.apply_mutation(self.pop[genes_to_mutate,:])
		# recombine existing population to make new individuals
		recombination_offset = self.rng.randint(low=1, high=len(self.pop))
		new_pop = self.apply_gene_crossover(self.pop, np.roll(self.pop, recombination_offset))
		self.pop = np.append(self.pop, new_pop, axis=0)[:self.pop_size,:]
		# if, somehow, we lost population, pad it with random samples
		if len(self.pop) < self.pop_size:
			self.pop=np.reshape(np.append(self.pop, self.rng.randint(low=0,high=256,size=[self.pop_size - len(self.pop), self.gene_bits])), [self.pop_size,self.gene_bits])

	def apply_mutation(self, genes):
		bits_to_swap = self.rng.choice(np.arange(self.gene_bits), np.int32(self.rng.normal(self.gene_bits/4)), replace=False)
		genes[:,bits_to_swap] = self.rng.randint(low=0, high=256, size=bits_to_swap.shape)
		return genes

	def apply_gene_crossover(self, gene1, gene2):
		# make sure we're combining the same shape gene samples
		assert np.shape(gene1[0]) == np.shape(gene2[0])
		crossover_point = np.int32(self.rng.normal(loc=self.gene_bits/2, scale=2, size=len(gene1)))
		# I'm not sure how to get a variable slice, so I'm doing it with a loop :(
		for i in range(len(gene1)):
			gene1[i], gene2[i] = np.append(gene1[i, crossover_point[i]:], gene2[i, :crossover_point[i]]), np.append(gene2[i, crossover_point[i]:], gene1[i, :crossover_point[i]])
		return np.append(gene1, gene2, axis=0)
	

	def sample_pop(self, nsamples):
		samples = np.int32(self.rng.choice(np.arange(self.pop_size), nsamples, replace=False))
		return self.pop[samples, :]