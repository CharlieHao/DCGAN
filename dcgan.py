#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

# Descriptions: DCGAN
# Structure:  1. Discriminator: leakyRelu, Generator: Relu + tanh
#         2. No fully connnected layer
#         3. Discriminator: stirded convolution (no pooling)
#          Generator: transposed stirded convolution                    
#           4. batch normalization

import os 
import util 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import scipy as sp 
from datetime import datetime
from dcgan_layers_tf import generate_sample_folder,lrelu,ConvLayer,FreactionallyStridedConvLayer,DenseLayer


# As description in the paper, preset hyperparameters 
LEARNING_RATE = 0.0002
BATCH_SIZE = 64
EPOCHS =5
BETA = 0.5

Save_sample_period = 50

generate_sample_folder()

class DCGAN(object):
	def __init__(self,img_length,num_colors,d_sizes,g_sizes):
		'''
		g_sizes,d_sizes are both dictionary used to store the structure of generator and discriminator
		'''
		self.img_length = img_length
		self.num_colors = num_colors
		self.latent_dims = g_sizes['z']

		# inputs
		self.X = tf.placeholder(
			tf.float32,
			shape = (None,img_length,img_length,num_colors),
			name = 'X'
		)
		self.Z = tf.placeholder(
			tf.float32,
			shape = (None,self.latent_dims),
			name = 'Z'
		)

		# discriminator
		logits = self.Discriminator(self.X,d_sizes)

		# generator
		self.sample_images = self.Generator(self.Z,g_sizes)

		# sample logits, principle of which is analogy to name space in C++
		# make use of tf.get_variable(<name>,<shape>,<type>,<initializer>)
		#         and tf.Variable_scope(<scope_name>)
		# In each name space:
		with tf.variable_scope('discriminator') as scope:
			scope.reuse_Variables()
			sample_logits = self.d_forward(self.sample_images,True)

		with tf.Variable_scope('generator') as scope:
			scope.reuse_Variables()
			self.sample_images_test = self.g_forward(
				self.Z, 
				reuse=True,
				is_training = False 
			)

		# cost function
		self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
			logits=logits,
			labels=tf.ones_like(logits)
 		)
		self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
			logits=sample_logits,
			labels=tf.zeros_like(sample_logits)
		)
		
		self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)
		self.g_cost = tf.reduce_mean(
			f.nn.sigmoid_cross_entropy_with_logits(
			logits=sample_logits,
			labels=tf.ones_like(sample_logits)
			)
		)
		real_predictions = tf.cast(logits > 0, tf.float32)
		fake_predictions = tf.cast(sample_logits < 0, tf.float32)
		num_predictions = 2.0*BATCH_SIZE
		num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
		self.d_accuracy = num_correct / num_predictions


		# optimizers
		self.d_params = [t for t in tf.trainable_variables() if t.name.startswith('d')]
		self.g_params = [t for t in tf.trainable_variables() if t.name.startswith('g')]

		self.d_train_op = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(
				self.d_cost, var_list=self.d_params
				)
		self.g_train_op = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(
				self.g_cost, var_list=self.g_params
				)

		self.init_op = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(self.init_op)


	def Discriminator(self, X, d_sizes):
		with tf.variable_scope("discriminator") as scope:
			# build convolution layers
			self.d_convlayers = []
			mi = self.num_colors
			dim = self.img_length
			count = 0
			for mo, filtersz, stride, apply_batch_norm in d_sizes['conv_layers']:
				# make up a name - used for get_variable
				name = "convlayer_%s" % count
				count += 1
				layer = ConvLayer(name, mi, mo, apply_batch_norm, filtersz, stride, lrelu)
				self.d_convlayers.append(layer)
			mi = mo
			print("dim:", dim)
			dim = int(np.ceil(float(dim) / stride))


		mi = mi * dim * dim
		
		# build dense layers
		self.d_denselayers = []
		for mo, apply_batch_norm in d_sizes['dense_layers']:
			name = "denselayer_%s" % count
			count += 1

			layer = DenseLayer(name, mi, mo, apply_batch_norm, lrelu)
			mi = mo
			self.d_denselayers.append(layer)
	
		# final logistic layer
		name = "denselayer_%s" % count
		self.d_finallayer = DenseLayer(name, mi, 1, False, lambda x: x)

		# get the logits
		logits = self.d_forward(X)

		return logits


	def d_forward(self, X, reuse=None, is_training=True):
		# encapsulate this because we use it twice
		output = X
		for layer in self.d_convlayers:
			output = layer.forward(output, reuse, is_training)
		output = tf.contrib.layers.flatten(output)
		for layer in self.d_denselayers:
			output = layer.forward(output, reuse, is_training)
		logits = self.d_finallayer.forward(output, reuse, is_training)
		return logits


	def Generator(self, Z, g_sizes):
		with tf.variable_scope("generator") as scope:
			# determine the size of the data at each step
			dims = [self.img_length]
			dim = self.img_length
			for _, _, stride, _ in reversed(g_sizes['conv_layers']):
				dim = int(np.ceil(float(dim) / stride))
				dims.append(dim)
			# note: dims is actually backwards
			# the first layer of the generator is actually last
			# so let's reverse it
			dims = list(reversed(dims))
			print("dims:", dims)
			self.g_dims = dims


			# dense layers
			mi = self.latent_dims
			self.g_denselayers = []
			count = 0
			for mo, apply_batch_norm in g_sizes['dense_layers']:
				name = "g_denselayer_%s" % count
				count += 1

				layer = DenseLayer(name, mi, mo, apply_batch_norm)
				self.g_denselayers.append(layer)
				mi = mo

			# final dense layer
			mo = g_sizes['projection'] * dims[0] * dims[0]
			name = "g_denselayer_%s" % count
			layer = DenseLayer(name, mi, mo, not g_sizes['bn_after_project'])
			self.g_denselayers.append(layer)


			# fs-conv layers
			mi = g_sizes['projection']
			self.g_convlayers = []

			# output may use tanh or sigmoid
			num_relus = len(g_sizes['conv_layers']) - 1
			activation_functions = [tf.nn.relu]*num_relus + [g_sizes['output_activation']]

			for i in range(len(g_sizes['conv_layers'])):
				name = "fs_convlayer_%s" % i
				mo, filtersz, stride, apply_batch_norm = g_sizes['conv_layers'][i]
				f = activation_functions[i]
				output_shape = [BATCH_SIZE, dims[i+1], dims[i+1], mo]
				print("mi:", mi, "mo:", mo, "outp shape:", output_shape)
				layer = FractionallyStridedConvLayer(
					name, mi, mo, output_shape, apply_batch_norm, filtersz, stride, f
				)
				self.g_convlayers.append(layer)
				mi = mo

			self.g_sizes = g_sizes
			return self.g_forward(Z)


	def g_forward(self, Z, reuse=None, is_training=True):		
		# dense layers
		output = Z
		for layer in self.g_denselayers:
			output = layer.forward(output, reuse, is_training)

		# project and reshape
		output = tf.reshape(
			output,
			[-1, self.g_dims[0], self.g_dims[0], self.g_sizes['projection']],
		)

		# apply batch norm
		if self.g_sizes['bn_after_project']:
			output = tf.contrib.layers.batch_norm(
				output,
				decay=0.9, 
				updates_collections=None,
				epsilon=1e-5,
				scale=True,
				is_training=is_training,
				reuse=reuse,
				scope='bn_after_project'
			)

		# pass through fs-conv layers
		for layer in self.g_convlayers:
			output = layer.forward(output, reuse, is_training)
	
		return output


	def fit(self, X):
		d_costs = []
		g_costs = []

		N = len(X)
		n_batches = N // BATCH_SIZE
		total_iters = 0
		for i in range(EPOCHS):
			print("epoch:", i)
			np.random.shuffle(X)
			for j in range(n_batches):
				t0 = datetime.now()

				if type(X[0]) is str:
				# is celeb dataset
					batch = util.files2images(
						X[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
					)

				else:
					# is mnist dataset
					batch = X[j*BATCH_SIZE:(j+1)*BATCH_SIZE]

				Z = np.random.uniform(-1, 1, size=(BATCH_SIZE, self.latent_dims))

				# train the discriminator
				_, d_cost, d_acc = self.sess.run(
					(self.d_train_op, self.d_cost, self.d_accuracy),
					feed_dict={self.X: batch, self.Z: Z},
				)				
				d_costs.append(d_cost)

				# train the generator
				_, g_cost1 = self.sess.run(
					(self.g_train_op, self.g_cost),
					feed_dict={self.Z: Z},
				)

				# g_costs.append(g_cost1)
				_, g_cost2 = self.sess.run(
					(self.g_train_op, self.g_cost),
					feed_dict={self.Z: Z},
				)
				g_costs.append((g_cost1 + g_cost2)/2) # just use the avg

				print("  batch: %d/%d  -  dt: %s - d_acc: %.2f" % (j+1, n_batches, datetime.now() - t0, d_acc))


				# save samples periodically
				total_iters += 1
				if total_iters % SAVE_SAMPLE_PERIOD == 0:
					print("saving a sample...")
					samples = self.sample(64) # shape is (64, D, D, color)

					# for convenience
					d = self.img_length
					if samples.shape[-1] == 1:
						# if color == 1, we want a 2-D image (N x N)
						samples = samples.reshape(64, d, d)
						flat_image = np.empty((8*d, 8*d))
						k = 0
						for i in range(8):
							for j in range(8):
								flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k].reshape(d, d)
								k += 1

					else:
						# if color == 3, we want a 3-D image (N x N x 3)
						flat_image = np.empty((8*d, 8*d, 3))
						k = 0
						for i in range(8):
							for j in range(8):
								flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k]
								k += 1
            
					sp.misc.imsave(
						'samples/samples_at_iter_%d.png' % total_iters,
						flat_image,
					)

		# save a plot of the costs
		plt.clf()
		plt.plot(d_costs, label='discriminator cost')
		plt.plot(g_costs, label='generator cost')
		plt.legend()
		plt.savefig('cost_vs_iteration.png')

	def sample(self, n):
		Z = np.random.uniform(-1, 1, size=(n, self.latent_dims))
		samples = self.sess.run(self.sample_images_test, feed_dict={self.Z: Z})
		return samples		











