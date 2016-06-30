# adversarial_networks.py
# Sam Goree
# Train one NN to generate images and another to classify it as "real" (from the training data) or "fake" (from the generative network)
# mlp and LogisticRegression are from Theano's deep learning tutorials, MNISTClassify is my own work following those tutorials

from MNISTClassify import ConvPoolingLayer, loadImages
from mlp import HiddenLayer
from LogisticRegression import *
import numpy as np
import matplotlib.pyplot as plt
from rmsprop import rmsprop
import scipy.misc

learning_rate = .0001
momentum = 0.9
batch_size = 500
num_epochs = 30
TRAIN_DIR = '/home/sam/Documents/mnist/TrainingImages/'
TEST_DIR = '/home/sam/Documents/mnist/TestImages/'

NUM_TRAINING_EXAMPLES = 40000 # must be less than 60000, that's all we have
NUM_VALIDATION_EXAMPLES = 20000
NUM_TEST_EXAMPLES = 0 # must be less than 10000

# load data
def load_data(dir):
	pass

# build & train

def buildAdv():
	rng = np.random.RandomState()

	print "Loading Data..."
	[training_data_shared, training_categories_shared, validation_data_shared, validation_categories_shared, test_data_shared, test_categories_shared] = loadImages(TRAIN_DIR, NUM_TRAINING_EXAMPLES, NUM_VALIDATION_EXAMPLES, TEST_DIR, NUM_TEST_EXAMPLES)
	current_generated_data_shared = theano.shared(np.zeros_like(training_data_shared.get_value()))
	current_generated_categories_shared = theano.shared(np.int32(np.zeros_like(training_categories_shared.get_value())))
	print "Building the model..."

	# we want a generative mlp that goes from a category and a random seed, through three hidden layers, to a 28x28 image
	xg_category = T.cast(T.ivector('xg_category'), 'int32') # input category
	xg_seed = T.matrix('xg_seed')

	xg_category_onehot = T.extra_ops.to_one_hot(xg_category, 10)

	# first mlp hidden layer """"""
	glayer0 = HiddenLayer(rng, input=T.concatenate((xg_category_onehot, xg_seed), axis=1), n_in=20, n_out=100, activation=T.tanh)
	# second mlp hidden layer
	glayer1 = HiddenLayer(rng, input=glayer0.output, n_in=100, n_out=500, activation=T.tanh)
	glayer2 = HiddenLayer(rng, input=glayer1.output, n_in=500, n_out=1000, activation=T.tanh)
	# last mlp hidden layer
	glayer3 = HiddenLayer(rng, input=glayer2.output, n_in=1000, n_out=28*28, activation=T.tanh)


	# the output of the generative network
	goutput = glayer3.output.reshape((batch_size, 1, 28,28))

	# we want a discriminative mlp that goes from an image, through a convolutional layer, through a hidden layer, through a logistic regression to decide whether 
	xd = glayer3.output # input images
	yd = xg_category # label for the images

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

	# This approach uses the negative log liklihood that the incorrect answer is selected (p_y_given_x is the softmax of the output of the layer)
	gcost = -T.mean(T.log(dlayer3.p_y_given_x)[T.arange(zd.shape[0]), 1-zd]) # using the given for zd

	dparams = dlayer3.params + dlayer2.params + dlayer1.params + dlayer0.params
	gparams = glayer3.params + glayer2.params + glayer1.params + glayer0.params

	dgrads = T.grad(dcost, dparams)
	ggrads = T.grad(gcost, gparams)

	dopt = rmsprop(dparams)
	gopt = rmsprop(gparams)
	dupdates = dopt.updates(dparams, dgrads,
                      learning_rate / np.cast['float32'](batch_size),
                      momentum)
	gupdates = gopt.updates(gparams, ggrads,
                      learning_rate / np.cast['float32'](batch_size),
                      momentum)

	# functions to train the discriminative model
	# important: call gfire first and set current_generated_data_shared and current_generated_categories_shared to the returned values before training
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

	dfire = theano.function([sample, category], dlayer3.p_y_given_x, givens={xd:sample, yd:category})


	# category values and random numbers - input to generative network
	ginput_categories_shared = theano.shared(np.int32(np.array(range(10)*(batch_size//10))))
	ginput_rng_shared = theano.shared(rng.uniform(low=-1,high=1, size=[batch_size,10]))
	
	# function to train the generative model
	gtrain_model = theano.function([], gcost, updates=gupdates, givens={xg_category:ginput_categories_shared, xg_seed:ginput_rng_shared, zd:theano.shared(np.int32(np.zeros(batch_size)))})

	# function to test/validate the generative model
	gtest_model = theano.function([], 1-dlayer3.errors(zd), givens={xg_category:ginput_categories_shared, xg_seed:ginput_rng_shared,zd:theano.shared(np.int32(np.zeros(batch_size)))})

	# function to generate new training examples for the discriminative model
	gfire = theano.function([], [goutput, xg_category], givens={xg_category:ginput_categories_shared, xg_seed:ginput_rng_shared})

	dbest_loss = 1

	def generate_new_data():
		ginput_rng_shared.set_value(rng.uniform(low=0,high=1, size=[batch_size,10]))
		generated_data = np.zeros([NUM_TRAINING_EXAMPLES,1,28,28])
		generated_categories = np.zeros_like(current_generated_categories_shared.get_value())
		for i in range(NUM_TRAINING_EXAMPLES//batch_size):
			generated_data[batch_size * i: batch_size * (i+1)], generated_categories[batch_size * i: batch_size * (i+1)] = gfire()
		current_generated_data_shared.set_value(generated_data.reshape([NUM_TRAINING_EXAMPLES,28*28]))
		current_generated_categories_shared.set_value(generated_categories)

	# training loop
	print("Training...")
	for epoch in range(1, num_epochs):
		print "training epoch: ", epoch
		if epoch % 10 == 0:
			print "evaluating discriminator..."
			# fire the generator before validating
			generate_new_data()
			# calculate validation loss
			dvalidation_loss = [dvalidate_model(i) for i in range(NUM_VALIDATION_EXAMPLES//batch_size)]
			mean_dvalidation_loss = np.mean(dvalidation_loss)
			print"discrimination error (% of real labeled fake): ", mean_dvalidation_loss * 100, "%"

			print "evaluating generator..."
			gvalidation_loss = gtest_model()
			print"generation quality (% of fake labeled real): ", (1-gvalidation_loss) * 100, "%"
			if epoch % 10 == 0:
				for i in range(10):
					#print "percent certainty that generated image is real: ", dfire(current_generated_data_shared[:batch_size].reshape([batch_size,28*28]),current_generated_categories_shared[:batch_size])[i]
					plt.imshow(np.reshape(current_generated_data_shared.get_value()[i], [28,28]), cmap=plt.get_cmap('gray'), interpolation='nearest')
					#plt.show()
					plt.savefig("output2/epoch" + str(epoch) + "image" + str(current_generated_categories_shared.get_value()[i]) + ".png")

		
		# fire the generator before training
		ginput_rng_shared = theano.shared(rng.uniform(low=0,high=1, size=[batch_size,10]))
		generate_new_data()
		dc = 1
		gc = 1
		for batch_index in range(NUM_TRAINING_EXAMPLES//batch_size * 2):
			if gc*.1 < dc :
				dc = dtrain_model(batch_index)
				if batch_index % (NUM_TRAINING_EXAMPLES//batch_size)  == 0: 
					print "dcost: ", dc
					test = dfire(np.append(training_data_shared.get_value()[0: batch_size//2],
																			 current_generated_data_shared.get_value()[0: batch_size//2], axis=0),
																			np.append(training_categories_shared.get_value()[0: batch_size//2], 
																			 current_generated_categories_shared.get_value()[0:batch_size//2], axis=0))
					print "percent certainty that real image is [fake,real]: ", test[0]
					print "correct answer: ", zd_given.get_value()[0]
					print "percent certainty that generated image is [fake,real]: ", test[batch_size//2]
					print "correct answer: ", zd_given.get_value()[batch_size//2]
			if dc*.1 < gc :
				ginput_rng_shared = theano.shared(rng.uniform(low=0,high=1, size=[batch_size,10]))
				gc = gtrain_model()
				if batch_index % (NUM_TRAINING_EXAMPLES//batch_size) == 0: print "gcost: ", gc
	
		
	plt.show()

	

buildAdv()