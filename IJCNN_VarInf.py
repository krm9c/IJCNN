# Time Series Simulation
# Author : Krishnan Raghavan
# Date: Dec 25, 2016
#######################################################################################
# Define all the libraries
import os, sys, random, time, tflearn
import numpy as np
from   sklearn import preprocessing
import tensorflow as tf

#######################################################################################
# Libraries created by us
# Use this path for Windows
# sys.path.append('C:\Users\krm9c\Dropbox\Work_9292016\Research\Common_Libraries')
# sys.path.append('C:\Users\krm9c\Desktop\Research\Paper_1_codes')
# path= "E:\Research_Krishnan\Data\Data_case_study_1"
# For MAC or Unix use this Path
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Common_Libraries')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_1_codes')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_2_codes')

# Set path for the data too
path = "/Users/krishnanraghavan/Documents/Data-case-study-1"
myRespath = "/Users/krishnanraghavan/DesktopRes/Results"
# Now import everything

from Library_Paper_two import *
from Library_Paper_one import Collect_samples_Bearing
###################################################################################
# Setup some parameters for the analysis
# Some global parameters for the script
# The NN parameters
Train_batch_size          = 100
Test_batch_size           = 100
Train_Glob_Iterations     = 50
Train_Loc_Iterations      = 50
# Start a time clock
start_time                = time.time()
# Set up parameters for dimension reduction
Faults                    = 4
###################################################################################
# Create a infinite Loop of Data-stream
def RollingDataImport(path, num):
	# Start_Analysis_Bearing()
	IR =  np.loadtxt('Results/Data/IR_sample.csv', delimiter=',')
	OR =  np.loadtxt('Results/Data/OR_sample.csv', delimiter=',')
	NL =  np.loadtxt('Results/Data/NL_sample.csv', delimiter=',')
	Norm =  np.loadtxt('Results/Data/Norm.csv'     , delimiter=',')
	sheet    = 'Test';
	f        = 'IR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_IR  =  np.array(import_data(filename,sheet, 1));
	sheet    = 'Test';
	f        = 'OR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_OR  =  np.array(import_data(filename,sheet, 1));
	sheet    = 'Test';
	f        = 'NL'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_NL  =  np.array(import_data(filename,sheet, 1));
	sheet    = 'normal';
	f        = 'Normal_1.xls'
	filename = os.path.join(path,f);
	Temp_Norm= np.array(import_data(filename,sheet, 1));
	return Temp_Norm, Temp_IR, Temp_OR, Temp_NL

###################################################################################
# Global Import of Data
Norm, IR, OR, NL = RollingDataImport(path, 1)
R  = np.array(OR)
R1 = np.array(NL)
R2 = np.array(Norm)
R3 = np.array(IR)
T  = np.concatenate((R, R1, R2, R3))
Y  = np.concatenate(((np.zeros((R.shape[0]))), (np.zeros((R1.shape[0]))+1), (np.zeros((R2.shape[0]))+2), (np.zeros((R3.shape[0]))+3) ))
def Inf_Loop_data(par):
	if par ==1:
		return (T + 0 * np.random.normal(0, 0.01, (T.shape[0], T.shape[1]))), tflearn.data_utils.to_categorical(Y, Faults)
	else:
		return (T + 0 * np.random.normal(0, 0.01, (T.shape[0], T.shape[1]))), tflearn.data_utils.to_categorical(Y, Faults)
####################################################################################
# Helper Function for the weight and the bias variable
# Weight
def weight_variable(shape, trainable, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, trainable = trainable, name = name)
# Bias function
def bias_variable(shape, trainable, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, trainable = trainable, name = name)
#  Summaries for the variables
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram_1', var)
# Class
class Agent():
    def __init__(self):
		self.classifier = {}
		self.Deep = {}
		self.Trainer = {}
		self.Summaries = {}
		self.sess = tf.InteractiveSession()

    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, key):
        with tf.name_scope(key):
            with tf.name_scope('weights'+key):
                self.classifier['Weight'+key] = weight_variable([input_dim, output_dim], trainable = trainability, name = 'Weight'+key)
                variable_summaries(self.classifier['Weight'+key])
            with tf.name_scope('bias'+key):
                self.classifier['Bias'+key] = bias_variable([output_dim], trainable = trainability, name = 'Bias'+key)
                variable_summaries(self.classifier['Weight'+key])
            with tf.name_scope('Wx_plus_b'+key):
                preactivate = tf.matmul(input_tensor, self.classifier['Weight'+key]) + self.classifier['Bias'+key]
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation'+key)
            tf.summary.histogram('activations', activations)
        return activations

    # Initialization for the default graph and the corresponding NN.
    def init_NN(self, R, Faults, lr, depth, Layers):
		Keys = []
		List = []
		with tf.name_scope("FLearners"):
			self.Deep['FL_layer0'] = tf.placeholder(tf.float32, shape=[None, R.shape[1]])
			for i in range(1,len(Layers)-1):
				self.Deep['FL_layer'+str(i)] = self.nn_layer(self.Deep['FL_layer'+str(i-1)], Layers[i-1], Layers[i], act=tf.nn.relu, trainability = False, key = 'FL_layer'+str(i))
				Keys.append(self.classifier['Weight'+'FL_layer'+str(i)])
				Keys.append(self.classifier['Bias'+'FL_layer'+str(i)])
			self.Deep['FL_layer'+str(len(Layers)-1)] = self.nn_layer( self.Deep['FL_layer'+str(i)], Layers[len(Layers)-2], Layers[len(Layers)-1], act=tf.nn.sigmoid, trainability = False, key = 'FL_layer'+str(len(Layers)-1))
			Keys.append(self.classifier['Weight'+'FL_layer'+str(len(Layers)-1)])
			Keys.append(self.classifier['Bias'+'FL_layer'+str(len(Layers)-1)])

		with tf.name_scope("Targets"):
			self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, Faults])

		with tf.name_scope("Classifier"):
			self.classifier['Fault'] = self.nn_layer( self.Deep['FL_layer'+str(len(Layers)-1)], Layers[len(Layers)-1], Faults, act=tf.identity, trainability =  False, key = 'Fault')
			tf.summary.histogram('Output', self.classifier['Fault'])
			List.append(self.classifier['WeightFault'])
			List.append(self.classifier['BiasFault'])

		with tf.name_scope("Trainer"):
			Error_Loss  =  tf.nn.softmax_cross_entropy_with_logits(logits = \
			 self.classifier['Fault'], \
			 labels = self.classifier['Target'], name=None)
			Latent_Loss =  0 # -2* tf.reduce_sum(tf.log(self.Deep['FL_layer'+str(len(Layers)-1)]))

			global_step = tf.Variable(0, trainable=False)
			LearningRate = tf.train.exponential_decay(lr, global_step, 100000, 0.99,\
			staircase=True)
			tf.summary.scalar('LearningRate', LearningRate)

			# Fault Learner
			self.Trainer["cost_NN"] =  tf.reduce_mean(Error_Loss)
			tf.summary.scalar('Cost_NN', self.Trainer["cost_NN"])
			self.Trainer['Optimizer_NN']    =  tf.train.GradientDescentOptimizer(LearningRate)
			self.Trainer["TrainStep_NN"]  =  self.Trainer['Optimizer_NN'].minimize(self.Trainer["cost_NN"], \
			global_step = global_step, var_list = List+Keys)

			with tf.name_scope('Evaluation'):
				with tf.name_scope('CorrectPrediction'):
					self.Trainer['correct_prediction'] = tf.equal(tf.argmax(self.classifier['Fault'],1), tf.argmax(self.classifier['Target'],1))
				with tf.name_scope('Accuracy'):
					self.Trainer['accuracy'] = tf.reduce_mean(tf.cast(self.Trainer['correct_prediction'], tf.float32))
				with tf.name_scope('Prob'):
					self.Trainer['prob'] = tf.cast((self.classifier['Target']),tf.float32)
				tf.summary.scalar('Accuracy', self.Trainer['accuracy'])
				tf.summary.histogram('Prob', self.Trainer['prob'])
		self.Summaries['merged'] = tf.summary.merge_all()
		self.Summaries['train_writer'] = tf.summary.FileWriter(myRespath + '/train', self.sess.graph)
		self.Summaries['test_writer'] = tf.summary.FileWriter(myRespath + '/test')
		self.sess.run(tf.global_variables_initializer())
		return self

#######################################################################
# Main Training Function for the model
def Training():
	# Generate the data and define parameter
	Xunscaled, Y =  Inf_Loop_data(1)
	scale = preprocessing.StandardScaler().fit(Xunscaled)
	X = scale.transform(Xunscaled)
	dimension = 4
	Ref, Tree = initialize_calculation(T = None, Data = Xunscaled, gsize = dimension, \
	par_train = 0)
	X, Tree = initialize_calculation(T = Tree, Data = Xunscaled, gsize = dimension,\
	par_train = 1)

	# Lets start with creating a model and then train batch wise.
	model = Agent()
	model = model.init_NN( X, Faults, 3e-4, 4, [X.shape[1], 200, 200, 200])

	# Declare a saver
	for i in range(Train_Glob_Iterations):
		P, Y = Inf_Loop_data(0)
		P = scale.transform(P)
		P, Tree = initialize_calculation(T = Tree, Data = P, gsize = dimension,\
		par_train = 1)
		print "Iteration == ", i
		for j in range(Train_Loc_Iterations):
			prev=0
			end=prev+Train_batch_size
			for k in range(((P.shape[0])/Train_batch_size)):
				batch_xs  = P[prev:end, :]
				batch_ys  = Y[prev:end, :]
				summary, _  = model.sess.run([model.Summaries['merged'], model.Trainer['TrainStep_NN']], feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
				if j % 10 == 0 :
						model.Summaries['train_writer'].add_summary(summary, j)
				prev = end
				end  = prev+Train_batch_size

		if i % 2 == 0:
			Sample, Y = Inf_Loop_data(0)
			Sample = scale.transform(Sample)
			Sample, Tree = initialize_calculation(T = Tree, Data = P, gsize = dimension,\
			par_train = 1)
			summary, acc  = model.sess.run( [model.Summaries['merged'], model.Trainer['accuracy']], feed_dict ={ model.Deep['FL_layer0']:Sample, model.classifier['Target']:Y })
			print "Accuracy--",i,"--", acc

if __name__ == "__main__":
	Training()
