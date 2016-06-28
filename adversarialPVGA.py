# adversarialGA.py
# identical to adversarialGA.py except uses ga_256 instead of ga

from MNISTClassify import ConvPoolingLayer, loadImages
from mlp import HiddenLayer
from LogisticRegression import *
import numpy as np
import matplotlib.pyplot as plt
from rmsprop import rmsprop
from PVGA import *

NUM_TRAINING_EXAMPLES = 20000 # must be less than 60000, that's all we have
NUM_VALIDATION_EXAMPLES = 10000
NUM_TEST_EXAMPLES = 0 # must be less than 10000
TRAIN_DIR = '/home/sam/Documents/mnist/TrainingImages/'
TEST_DIR = '/home/sam/Documents/mnist/TestImages/'

learning_rate = .0001
momentum = 0.9
batch_size = 1000
num_epochs = 250
num_generations_per_epoch = 20

digit_to_generate = 1


def buildAdversarialGA():
	rng = np.random.RandomState()

	print "Loading Data..."
	[training_data_shared, training_categories_shared, validation_data_shared, validation_categories_shared, test_data_shared, test_categories_shared] = loadImages(TRAIN_DIR, NUM_TRAINING_EXAMPLES, NUM_VALIDATION_EXAMPLES, TEST_DIR, NUM_TEST_EXAMPLES)
	current_generated_data_shared = theano.shared(np.zeros_like(training_data_shared.get_value()))
	current_generated_categories_shared = theano.shared(np.int32(np.zeros_like(training_categories_shared.get_value())))
	print "Building the model..."

	
	xd = T.matrix('xd') # input images
	yd = T.ivector('yd') # label for the images

	yd_onehot = T.extra_ops.to_one_hot(yd, 10)

	zd = T.ivector('z') # real or not

	dlayer0Input = xd.reshape((batch_size, 1, 28, 28)) # reassemble the image

	# convolutional layer
	dlayer0 = ConvPoolingLayer(rng, input=dlayer0Input, image_shape=(batch_size, 1, 28, 28), filter_shape=(20, 1, 5, 5), poolsize=(2,2))
	# first mlp hidden layer
	dlayer1 = HiddenLayer(rng, input=dlayer0.output.flatten(2), n_in=20*9*4*4, n_out=500, activation=T.tanh)
	# second mlp hidden layer
	dlayer2 = HiddenLayer(rng, input=T.concatenate((dlayer1.output, yd_onehot), axis=1), n_in=510, n_out=100, activation=T.tanh)
	# logistic regression layer - note the dimension magic needed to append the category to each input vector
	dlayer3 = LogisticRegression(input=dlayer2.output, n_in=100, n_out=2) # 0 is generated, 1 is real

	dcost = dlayer3.negative_log_likelihood(zd)

	dparams = dlayer3.params + dlayer2.params + dlayer1.params + dlayer0.params

	dgrads = T.grad(dcost, dparams)

	dopt = rmsprop(dparams)

	dupdates = dopt.updates(dparams, dgrads,
                      learning_rate / np.cast['float32'](batch_size),
                      momentum)

	zd_temp = np.zeros(batch_size)
	zd_temp[:batch_size//2] = 1
	zd_given = theano.shared(np.int32(zd_temp), borrow=True)

	index = T.lscalar()

	# training occurs on both real and generated training data
	dtrain_model = theano.function([index], dcost, updates=dupdates, givens={xd:T.concatenate((training_data_shared[index * batch_size//2: (index+1) * batch_size//2],
																			 current_generated_data_shared[index * batch_size//2: (index+1) * batch_size//2])), 
																			 yd:T.concatenate((training_categories_shared[index * batch_size//2: (index+1) * batch_size//2], 
																			 current_generated_categories_shared[index * batch_size//2: (index+1)* batch_size//2])),
																			 zd:zd_given})
	# validation is just on the validation data - it sees how many of them it thinks are real
	dvalidate_model = theano.function([index], dlayer3.errors(zd), givens={xd:validation_data_shared[index * batch_size: (index+1) * batch_size],
																			 yd:validation_categories_shared[index * batch_size: (index+1) * batch_size],
																			 zd:theano.shared(np.int32(np.ones(batch_size)))})
	# testing occurs on real and generated
	dtest_model = theano.function([index], dlayer3.errors(zd), givens={xd:T.concatenate((test_data_shared[index * batch_size//2: (index+1) * batch_size//2],
																			 current_generated_data_shared[index * batch_size//2: (index+1) * batch_size//2])), 
																			 yd:T.concatenate((test_categories_shared[index * batch_size//2: (index+1) * batch_size//2], 
																			 current_generated_categories_shared[index * batch_size//2: (index+1)* batch_size//2])),
																			 zd:zd_given})
	sample = T.matrix('samples')
	category = T.ivector('category')
	real = T.ivector('real?')
	# fire the discriminator on input samples, what category they are trying to be and whether they're real or generated (1 == real)
	dfire = theano.function([sample, category], dlayer3.p_y_given_x, givens={xd:sample, yd:category})
	# fitness function needs to accomidate a list of genes - currently it only accepts batch_size at once, so it loops
	def fitness_function(a):
		retval = np.array([])
		for i in range(len(a)//batch_size):
			retval = np.append(retval, dfire(np.packbits(a, axis=1)[i*batch_size: (i+1) * batch_size, :], [digit_to_generate]*batch_size)[:,1])
		return retval
	current_generated_categories_shared.set_value([digit_to_generate]*NUM_TRAINING_EXAMPLES)



	gmodel = pvga(fitness_function, 28*28*8, rng, num_samples_per_gen=2000)


	mutation_rate = .05
	print "Training..."

	for epoch in range(1, num_epochs):
		print "Epoch: ", epoch
		if epoch %10 == 0:
			#print "Validating..."
			#error = np.array([])
			#for batch_index in range(NUM_VALIDATION_EXAMPLES//batch_size *2):
				#error = np.append(error, dvalidate_model(batch_index))
			#print "D error: ", np.mean(error) * 100, "%"
			plt.imshow(np.reshape(current_generated_data_shared.get_value()[0],[28,28]), cmap=plt.get_cmap('gray'), interpolation='nearest')
			plt.draw()
		data = np.packbits(gmodel.sample_pop(NUM_TRAINING_EXAMPLES),axis=1)
		current_generated_data_shared.set_value(data.reshape((NUM_TRAINING_EXAMPLES, 28*28)))
		print "Training D:"
		loss = np.array([])
		for batch_index in range(NUM_TRAINING_EXAMPLES//batch_size * 2):
			loss = np.append(loss, dtrain_model(batch_index))
		print "Average loss: ", np.mean(loss)
		print "Training G:"
		gmodel.create_next_generation(mutation_rate=mutation_rate)
		prev_max = gmodel.max_fitness
		for i in range(epoch * 50):
			#mutation_rate/=2
			print "Max fitness: ", gmodel.max_fitness
			gmodel.create_next_generation(mutation_rate=mutation_rate)
			prev_max = gmodel.max_fitness
		plt.show()

buildAdversarialGA()