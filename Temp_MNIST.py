# Deep Learning Simulations
# Author : Krishnan Raghavan
# Date: Dec 25, 2016
#######################################################################################
# Define all the libraries
import os, sys, random, time, tflearn
import numpy as np
from   sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
### Import all the libraries required by us
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Common_Libraries')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_1_codes')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_2_codes')
from Library_Paper_two import *
from Data_import import *
## Set path for the data
path = "/Users/krishnanraghavan/Documents/Data-case-study-1"
myRespath = "/Users/krishnanraghavan/DesktopRes/Results"
###################################################################################
# Setup some parameters for the analysis
# The NN parameters
Train_batch_size = 100
Test_batch_size = 100
Train_Glob_Iterations = 500
####################################################################################
# Helper Function for the weight and the bias variable
# Weight
def weight_variable(shape, trainable, name):
  initial = tf.truncated_normal(shape, stddev=1)
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
    ###################################################################################
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
    ###################################################################################
    # Initialization for the default graph and the corresponding NN.
    def init_NN(self, R, Faults, lr, Layers):
        Keys = []
        with tf.name_scope("FLearners"):
            self.Deep['FL_layer0'] = tf.placeholder(tf.float32, shape=[None, R.shape[1]])
            for i in range(1,len(Layers)):
                self.Deep['FL_layer'+str(i)] = self.nn_layer(self.Deep['FL_layer'+str(i-1)], Layers[i-1],\
                Layers[i], act=tf.nn.tanh, trainability = False, key = 'FL_layer'+str(i))
                Keys.append(self.classifier['Weight'+'FL_layer'+str(i)])
                Keys.append(self.classifier['Bias'+'FL_layer'+str(i)])
        with tf.name_scope("Targets"):
            self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, Faults])
        with tf.name_scope("Classifier"):
            self.classifier['Fault'] = self.nn_layer( self.Deep['FL_layer'+str(len(Layers)-1)],\
            Layers[len(Layers)-1], Faults, act=tf.identity, trainability =  False, key = 'Fault')
            tf.summary.histogram('Output', self.classifier['Fault'])
            Keys.append(self.classifier['WeightFault'])
            Keys.append(self.classifier['BiasFault'])
        with tf.name_scope("Trainer"):
            Error_Loss  =  tf.nn.softmax_cross_entropy_with_logits(logits = \
            self.classifier['Fault'], \
            labels = self.classifier['Target'], name=None)
            global_step = tf.Variable(0, trainable=False)
            LearningRate = tf.train.exponential_decay(lr, global_step, 100000, 0.99,\
			staircase=True)
            Reg = tf.add_n([ tf.nn.l2_loss(v) for v in Keys ]) * 0.001
            tf.summary.scalar('LearningRate', LearningRate)
            # Fault Learner
            self.Trainer["cost_NN"] =  tf.reduce_mean(Error_Loss+ Reg)
            tf.summary.scalar('Cost_NN', self.Trainer["cost_NN"])
            self.Trainer['Optimizer_NN']    =  tf.train.GradientDescentOptimizer(LearningRate)
            self.Trainer["TrainStep_NN"]  =  self.Trainer['Optimizer_NN'].minimize(self.Trainer["cost_NN"], \
			global_step = global_step, var_list = Keys)
            with tf.name_scope('Evaluation'):
                with tf.name_scope('CorrectPrediction'):
                    self.Trainer['correct_prediction'] = tf.equal(tf.argmax(self.classifier['Fault'],1),\
                    tf.argmax(self.classifier['Target'],1))
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
# Set up parameters for dimension reduction
###################################################################################
# Let us import some dataset
# First up is the Rolling Element Bearing Data-set
# Syntax -- 1
# DataImport(num, classes=4, file=0, sample_size = 1000, features = 100)
# Syntax -- 2
# initialize_calculation(T, Data, gsize, par_train)
def PreProcess(num, dimension, flag):
    T, Y = DataImport(num=num, sample_size = 4000, features = 1000)
    if flag ==0:
        Ref, Tree = initialize_calculation(T = None, Data = T, gsize = dimension,\
        par_train = 0)
        X, Tree = initialize_calculation(T = Tree, Data = T, gsize = dimension,\
        par_train = 1)
        return X, Tree, Y
    else:
        return T, Y
###################################################################################
# Let us import some dataset
# Lets do the analysis
def Analyse(TrainX, TestX, TrainY, TestY):
    import gc
    # Lets start with creating a model and then train batch wise.
    model = Agent()
    model = model.init_NN(TrainX, 4, 3e-6, [TrainX.shape[1], 1024, 1024, 1024])
    ####################################################################################
    ## Start the learning Procedure Now
    acc = []
    # Declare a saver
    for i in tqdm(xrange(Train_Glob_Iterations)):
        for k in tqdm(xrange(((TrainX.shape[0])/Train_batch_size))):
            rand= [ random.randint(0, TrainX.shape[0]-1) for o in xrange(Train_batch_size) ]
            batch_xs  = TrainX[rand, :]
            batch_ys  = TrainY[rand, :]
            summary, _  = model.sess.run([model.Summaries['merged'], model.Trainer['TrainStep_NN']],\
             feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
        if i % 1 == 0:
            summary, a  = model.sess.run( [model.Summaries['merged'], model.Trainer['accuracy']], feed_dict\
             ={ model.Deep['FL_layer0'] : TestX, model.classifier['Target'] : TestY})
            acc.append(a)
        if a > 0.99:
            break
    tf.reset_default_graph()
    del model
    gc.collect()
    return acc


# DataSet -- 1
print "----------------------------Rolling Element Bearing--------------------------------------"
AccW = []
AccWO = []
for x in tqdm(xrange(1)):
    num =2
    # Method -- 1 # With the pre processing step
    X, Y = PreProcess(num, 4, 0)
    P = preprocessing.scale(X)
    TrainX, TestX, TrainY, TestY = train_test_split(P, Y, test_size = 0.70)
    acc = Analyse(TrainX, TestX, TrainY, TestY)
np.savetxt('ConvergenceWMNIST.csv', acc, delimiter =',' )

# # Case --5
# num = 0
# # Dataset --1
# # Method -- 1 # Without the pre processing step
# # Let get the data into a folder
# clear = lambda: os.system('clear') #on Linux System
# clear()
# X, Y = PreProcess(num, 4, 1)
# P = preprocessing.scale(X)
# TrainX, TestX, TrainY, TestY = train_test_split(P, Y, test_size = 0.50)
# acc = Analyse(TrainX, TestX, TrainY, TestY)
# print "Accuracy--11", acc
#
# # Case --6
# num = 0
# # Dataset --1
# # Method -- 2 # With the pre processing step
# # Let get the data into a folder
# X, Tree, Y = PreProcess(num, 4, 0)
# P = preprocessing.scale(X)
# TrainX, TestX, TrainY, TestY = train_test_split(P, Y, test_size = 0.50)
# acc = Analyse(TrainX, TestX, TrainY, TestY)
# print "Accuracy--12", acc
# # Case --4
# # Dataset --2
# # Method -- 2 # With the pre processing step
# # Let get the data into a folder
# num =1
# X, Tree, Y = PreProcess(num, 4, 0)
# P = preprocessing.scale(X)
# TrainX, TestX, TrainY, TestY = train_test_split(P, Y, test_size = 0.50)
# acc = Analyse(TrainX, TestX, TrainY, TestY)
# print "Accuracy--22", acc
#
# # Case --5
# num = 0
# # Dataset --1
# # Method -- 1 # Without the pre processing step
# # Let get the data into a folder
# clear = lambda: os.system('clear') #on Linux System
# clear()
# X, Y = PreProcess(num, 4, 1)
# P = preprocessing.scale(X)
# TrainX, TestX, TrainY, TestY = train_test_split(P, Y, test_size = 0.50)
# acc = Analyse(TrainX, TestX, TrainY, TestY)
# print "Accuracy--11", acc
#
# # Case --6
# num = 0
# # Dataset --1
# # Method -- 2 # With the pre processing step
# # Let get the data into a folder
# X, Tree, Y = PreProcess(num, 4, 0)
# P = preprocessing.scale(X)
# TrainX, TestX, TrainY, TestY = train_test_split(P, Y, test_size = 0.50)
# acc = Analyse(TrainX, TestX, TrainY, TestY)
# print "Accuracy--12", acc
