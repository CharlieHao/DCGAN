#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

# Descriptions: DCGAN
# Structure:	1. Discriminator: leakyRelu, Generator: Relu + tanh
# 				2. No fully connnected layer
#		  		3. Discriminator: stirded convolution (no pooling)
#				   Generator: transposed stirded convolution						 			  
# 		  		4. batch normalization
#  

import os 
import util
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import scipy as sp 
from datetime import datetime
from tensorflow.contrib.layers import batch_norm


def generate_sample_folder():
	if not os.path.exists('samples'):
		os.mkdir('samples')

def lrelu(x,slope=0.2):
	return np.maximum(x,slope*x)

class ConvLayer(object):
	def __init__(self,name,mi,mo,batch_mormalization=True,filter_size=5,stride=2,activation=tf.nn.relu):
		self.W = tf.get_variable(
			'W_%s'%name,
			shape = (filter_size,filter_size,mi,mo),
			initializer = tf.truncated_normal_initializer(stddev=0.02),
		)
		self.b = tf.get_variable(
			'b_%s'%name,
			shape = (mo,),
			initializer = tf.zeros_initializer(),
		)
		self.name = name
		self.activation = activation
		self.batch_mormalization = batch_mormalization
		self.stride = stride
		self.params = [self.W, self.b]

	def forward(self,x,reuse,is_training):
		'''
		reuse: Since there are two steps of gradient: Discriinator and Generator
		is_training: parameters in batch normalization is different in training process and test process
		'''
		outcome = tf.nn.conv2d(
			x,
			self.W,
			strides = [1,self.stride,self.stride,1],
			padding = 'SAME'
		)
		outcome = tf.nn.bias_add(outcome,self.b)

		# whether to use batch nomalization and training or testing
		if self.batch_mormalization:
			outcome = batch_norm(
				outcome,
				decay=0.9,
				updates_collection=None,
				epsilon=1e-5,
				scale=True,
				is_training=is_training,
				reuse=reuse,
				scope=self.name,
			)
		return self.activation(outcome)

class FreactionallyStridedConvLayer(object):
	def __init__(self,name,mi,mo,output_shape,batch_nomalization=True,filter_size=5,stride=2,activation=tf.nn.relu):
		# based on the behavior of transpose conv, exchange the position of mo and mi
		self.W = tf.get_value(
			'W_%s'%name,
			shape = (filter_size,filter_size,mo,mi), # In conv2d_transpose: known output, find input, so exchange the position of these two 
			initializer = tf.random_normal_initializer(stddev=.002)
		)
		self.b = tf.get_variable(
			'b_%s'%name,
			shape = (mo,),
			initializer = tf.zeros_initializer()
		)
		self.name = name
		self.output_shape = output_shape
		self.batch_mormalization = batch_mormalization
		self.filter_size = filter_size
		self.stride = stride
		self.activation = activation	

	def forward(self,x,reuse,is_training):
		outcome = tf.nn.conv2d_transpose(
			value=x,
			filter = self.W,
			output_shape = self.output_shape,
			strides = [1,self.stride,self.stride,1]
		)
		outcome = tf.nn.bias_add(outcome,self.b)
		
		# whether t ouse batch normalization and training or testing 
		if self.batch_mormalization:
			outcome = batch_norm(
				outcome,
				decay=0.9,
				updates_collection=None,
				epsilon=1e-5,
				scale=True,
				is_training=is_training,
				reuse=reuse,
				scope=self.name,
			)
			return self.activation(outcome)

class DenseLayer(object):
	def __init__(self, name, M1, M2, apply_batch_norm, activation=tf.nn.relu):
		self.W = tf.get_variable(
			"W_%s" % name,
			shape=(M1, M2),
			initializer=tf.random_normal_initializer(stddev=0.02),
		)
		self.b = tf.get_variable(
			"b_%s" % name,
			shape=(M2,),
			initializer=tf.zeros_initializer(),
		)
		self.activation  = activation	
		self.name = name
		self.apply_batch_norm = apply_batch_norm
		self.params = [self.W, self.b]

	def forward(self, X, reuse, is_training):
		outcome = tf.matmul(X, self.W) + self.b

		# apply batch normalization
		if self.apply_batch_norm:
			outcome = tf.contrib.layers.batch_norm(
				outcome,
				decay=0.9, 
				updates_collections=None,
				epsilon=1e-5,
				scale=True,
				is_training=is_training,
				reuse=reuse,
				scope=self.name,
		)
		return self.activation(outcome)




