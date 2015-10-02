from scipy import io
import scipy.special as sigmoid
import numpy
import random
import time
import csv
import matplotlib.pyplot as plt

#Flag to determine whether or not to run error rates every 5000 iterations
flag = True


#Import data
data = io.loadmat("./digit-dataset/train.mat")
test = io.loadmat("./digit-dataset/test.mat")

#Unravel and organize data
images = numpy.array(data["train_images"])
training = numpy.array([[images[:,:,x]] for x in xrange(0,60000)])
training = numpy.array([training[x].flatten() for x in xrange(0,60000)])

labels = numpy.array([data["train_labels"][x][0] for x in xrange(0,60000)])

test_images = numpy.array(test["test_images"])
test_set = numpy.array([[test_images[:,:,x]] for x in xrange(0,10000)])
test_set = numpy.array([test_set[x].flatten() for x in xrange(0,10000)])

#"Normalize" images
training = 1/255. * training
test_set = 1/255. * test_set

#X is the input before weight calculation. delta is the backwards propagation value
#x and delta must be numpy arrays
def update_function(weight, step, x, delta):
	# Make sure this multiplies delta, x correctly. Want to matrix multiply, not dot
	return weight - step * numpy.dot(numpy.vstack(x), numpy.vstack(delta).transpose())

def MSE_delta(true_label, predict_label):
	#Want to dot each entry, not matrix multiply (!BE CAREFUL)
	first = true_label - predict_label
	second = predict_label * (numpy.vstack(numpy.ones(10)) - predict_label)
	return -1 * first * second

def CEE_delta(true_label, predict_label):
	return predict_label - true_label

def forward_pass(weight_hidden, weight_out, image):
	#Returns a dictionary of lists of the outputs. {input: (785,1),hidden: (201,1),output: (10,1)}
	outputs = {}
	outputs['input'] = numpy.vstack(image)
	s_hidden = numpy.dot(outputs['input'].transpose(), weight_hidden)
	x_hidden = numpy.tanh(s_hidden)
	x_hidden = numpy.insert(x_hidden, 0, 1)
	outputs['hidden'] = numpy.vstack(x_hidden)
	s_out = numpy.dot(outputs['hidden'].transpose(), weight_out)
	x_out = sigmoid.expit(s_out)
	outputs['output'] = numpy.vstack(x_out).transpose()
	return outputs

def backward_pass(outputs, true_label, weight_out, loss_fn):
	#Returns a dictionary of delta values
	deltas = {}
	predict_label = outputs['output']
	deltas['output'] = loss_fn(true_label, predict_label)
	x_hidden = outputs['hidden'][1:]
	first = numpy.vstack(numpy.ones(200)) - x_hidden * x_hidden
	repeat_deltas = numpy.repeat(deltas['output'].transpose(), 201, axis = 0)
	second =  numpy.sum(weight_out * repeat_deltas, axis=1)
	second = numpy.vstack(second[1:])
	deltas['hidden'] = first * second
	return deltas

def trainNeuralNetwork(images, label, loss_fn, step = 0.1):
	iterations = []
	error_rates = []
	#Should be size(785, 200). Randomly initiated values on a normal distribution (0,1)
	w_hidden = numpy.random.randn(785,200) * 0.0001
	#Should be size(201, 10). Randomly initiated values on a normal distribution (0,1)
	w_out = numpy.random.randn(201,10) * 0.0001
	iteration = 0
	while iteration < 70000:
		if (iteration % 5000 == 0) and flag:
			print iteration 
			error = 0
			for x in xrange(0,60000):
				if predictNeuralNetwork(w_hidden,w_out,images[x]) != label[x]:
					error += 1
			error_rates.append(error/60000.)
			iterations.append(iteration)
		#Random choice of data point
		index = random.randint(0,59999)
		data_point = images[index]
		#bias point
		data_point = numpy.insert(data_point, 0, 1)
		#forward pass
		outputs = forward_pass(w_hidden, w_out, data_point)
		true_label = numpy.zeros(10)
		true_label[label[index]] = 1.
		#backward pass
		deltas = backward_pass(outputs, numpy.vstack(true_label), w_out, loss_fn)
		#update
		w_hidden = update_function(w_hidden, step, outputs['input'], deltas['hidden'])
		w_out = update_function(w_out, step, outputs['hidden'], deltas['output'])
		#increment iteration
		iteration += 1
	ax = plt.gca()
	ax.get_yaxis().get_major_formatter().set_useOffset(False)
	ax.get_yaxis().get_major_formatter().set_scientific(False)
	plt.plot(iterations, error_rates)
	plt.ylabel('Training Error')
	plt.xlabel('Iterations')
	if step == 0.1:
		plt.title('Training Error Using MSE')
		plt.savefig('mseplt')
	else:
		plt.title('Training Error Using Cross Entropy')
		plt.savefig('ceeplt')
	plt.clf()
	return w_hidden, w_out


def predictNeuralNetwork(weight_hidden, weight_out, image):
	#Insert bias
	vector = numpy.insert(image, 0, 1)
	#Hidden kayer
	s_hidden = numpy.dot(numpy.vstack(vector).transpose(), weight_hidden)
	x_hidden = numpy.tanh(s_hidden)
	#Insert bias
	x_hidden = numpy.vstack(numpy.insert(x_hidden, 0, 1)).transpose()
	#Output layer
	s_out = numpy.dot(x_hidden, weight_out)
	x_out = sigmoid.expit(s_out)
	#Should be of shape (1,10) 
	return numpy.argmax(x_out)

index = [x for x in xrange(0,60000)]
random.shuffle(index)

train_set = numpy.array([training[index[x]] for x in xrange(0,50000)])
label_set = numpy.array([labels[index[x]] for x in xrange(0,50000)])

start = time.time()
w1, w2 = trainNeuralNetwork(training, labels, MSE_delta)
print("--- %s seconds to finish training using MSE ---" % (time.time() - start))
"""
error = 0
for x in xrange(50000,60000):
	if predictNeuralNetwork(w1,w2,training[index[x]]) != labels[index[x]]:
		error += 1
print "Error Rate MSE:"
print error/10000.
"""
start = time.time()
w3, w4 = trainNeuralNetwork(training, labels, CEE_delta, step = 0.025)
print("--- %s seconds to finish training using Cross Entropy ---" % (time.time() - start))
"""
error = 0
for x in xrange(50000,60000):
	if predictNeuralNetwork(w3,w4,training[index[x]]) != labels[index[x]]:
		error += 1
print "Error Rate CEE:"
print error/10000.
"""
#Write to csvfile
count = 1
with open('neuralnets.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Id', 'Category'])
	for x in xrange(0,len(test_set)):
		writer.writerow([count, predictNeuralNetwork(w3,w4,test_set[x])])
		count += 1
