# MNISTClassify.py
# Classifies mnist training images using a convolutional NN in theano
# based heavily on tutorial at http://deeplearning.net/tutorial/lenet.html

from PIL import Image
from os import listdir
import sys
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import matplotlib.pyplot as plt
from theano.tensor.signal import downsample

from LogisticRegression import *
from mlp import HiddenLayer


TRAIN_DIR = '/home/sam/Documents/mnist/TrainingImages/'
TEST_DIR = '/home/sam/Documents/mnist/TestImages/'

NUM_TRAINING_EXAMPLES = 40000 # must be less than 60000, that's all we have
NUM_VALIDATION_EXAMPLES = 20000
NUM_TEST_EXAMPLES = 10000 # must be less than 10000

N_EPOCHS = 200
learning_rate = 0.1

class ConvPoolingLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input




# load images from a directory
def loadImages(trainDir, numTrainingExamples, numValidationExamples, testDir, numTestExamples):
	image_filenames = map((lambda s: trainDir + s), listdir(trainDir)[:numTrainingExamples+numValidationExamples])
	training_data = np.zeros([len(image_filenames), 28*28])
	training_categories = np.zeros([len(image_filenames)])
	i = 0
	for f in image_filenames:
		im = Image.open(f)
		training_data[i] = np.asarray(im)[:,:,0].flatten()
		training_categories[i] = int(image_filenames[i][-5])
		if i%1000 == 0: print "Loaded file: ", i
		i += 1
		
	image_filenames = map((lambda s: testDir + s), listdir(TEST_DIR)[:numTestExamples])
	test_data = np.zeros([len(image_filenames), 28*28])
	test_categories = np.zeros(len(image_filenames))
	i = 0
	for f in image_filenames:
		im = Image.open(f)
		test_data[i] = np.asarray(im)[:,:,0].flatten()
		test_categories[i] = int(image_filenames[i][-5])
		if i%1000 == 0: print"Loaded file: ", i
		i += 1
	# make them into theano shared variables for use later
	training_data_shared = theano.shared(training_data[:numTrainingExamples], borrow=True)
	training_categories_shared = theano.shared(np.int32(training_categories[:numTrainingExamples]), borrow=True)
	validation_data_shared = theano.shared(training_data[numTrainingExamples:], borrow=True)
	validation_categories_shared = theano.shared(np.int32(training_categories[numTrainingExamples:]), borrow=True)
	test_data_shared = theano.shared(test_data, borrow=True)
	test_categories_shared = theano.shared(np.int32(test_categories), borrow=True)
	return [training_data_shared, training_categories_shared, validation_data_shared, validation_categories_shared, test_data_shared, test_categories_shared]
def classify():
	[training_data_shared, training_categories_shared, 
	validation_data_shared, validation_categories_shared,
	test_data_shared, test_categories_shared] = loadImages(TRAIN_DIR, NUM_TRAINING_EXAMPLES, NUM_VALIDATION_EXAMPLES, TEST_DIR, NUM_TEST_EXAMPLES)
	# create a nn with two 28**2 5x5 convolutional layer (each node looks at 25 pixels in a square) + convolutional pooling layer, then a regular nn layer

	print"building the model..."

	rng = np.random.RandomState()
	nkerns=[20,50]
	batch_size = 100
	n_train_batches = NUM_TRAINING_EXAMPLES/batch_size

	x = T.matrix('x')
	y = T.ivector('y')

	index = T.lscalar()

	layer0_input = x.reshape((batch_size, 1, 28, 28))

	layer0 = ConvPoolingLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 28, 28), filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2,2))
	# outputs a convoluted image
	layer1 = ConvPoolingLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], 12, 12), filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2,2))
	# outputs a doubly convoluted image
	# flatten before putting it through the rest of the NN
	layer2_input = layer1.output.flatten(2)
	layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1]*4*4, n_out=500, activation=T.tanh) # tanh activation function (sigmoidal)

	layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

	cost = layer3.negative_log_likelihood(y)

	test_model = theano.function([index], layer3.errors(y), givens={x:test_data_shared[index * batch_size: (index+1) * batch_size],
	                                                                y:test_categories_shared[index * batch_size: (index+1) * batch_size]})
	validate_model = theano.function([index], layer3.errors(y), givens={x:validation_data_shared[index * batch_size: (index+1) * batch_size],
	                                                                    y:validation_categories_shared[index * batch_size: (index+1) * batch_size]})

	# train on the images

	print"training..."

	params = layer3.params + layer2.params + layer1.params + layer0.params
	grads = T.grad(cost, params) # the magic step

	updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

	train_model = theano.function([index], cost, updates=updates, givens={x:training_data_shared[index * batch_size: (index+1) * batch_size],
	                                                                             y:training_categories_shared[index * batch_size: (index+1) * batch_size]})
	epoch = 0

	best_loss = 1

	while (epoch < N_EPOCHS):
		epoch +=1
		if epoch %5 == 0: print"training epoch: ", epoch
		if epoch %5 == 0:
				print"validating..."
				validation_loss = [validate_model(i) for i in range(NUM_VALIDATION_EXAMPLES/batch_size)]
				this_validation_loss = np.mean(validation_loss)
				if this_validation_loss < best_loss:
					best_loss = this_validation_loss
				else:
					print "loss is worse now, breaking..."
					break
				print"loss: ", this_validation_loss * 100, "%"
		for minibatch_index in range(n_train_batches):
			
			train_model(minibatch_index)
			
	print"done training! Testing..."
	test_losses = [test_model(i) for i in range(NUM_TEST_EXAMPLES/batch_size)]
	test_score = np.mean(test_losses)
	print"Test score loss: ", test_score * 100, "%"
if __name__ == "__main__": classify()