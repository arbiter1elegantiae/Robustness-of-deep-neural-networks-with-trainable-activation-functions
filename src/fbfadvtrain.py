import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

from art.utils import random_sphere

class FastBetterFreeTrainedModel(Model):

	# Add epsilon attribute
	def __init__(self, eps = 8/255, **kwargs):
		super(FastBetterFreeTrainedModel, self).__init__(**kwargs)
		self.eps = eps


	def loss_gradient_delta_framework(self, x, y, delta):
		"""
		Compute the gradient of the loss function w.r.t. `delta`.
		:param x: Input with shape as expected by the model. 
		:param y: Indices of shape (nb_samples,).
		:param delta: Input with shape as expected from the model
		:return: Gradients of the same shape as `x`.
		"""
		if self.compiled_loss is None:
			raise ValueError("Loss object is necessary for computing the loss gradient.")

		with tf.GradientTape() as tape:
			tape.watch(delta)
			y_pred = self(x + delta, training=True)
			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
			
		return tape.gradient(loss, delta)
	
	def fsgm_rand_delta(self, dim_x, dim_y, x_batch, y_batch):
		# Set a random init for delta w/ L_inf = 8/255 and compute gradients wrt x+delta
		delta = tf.reshape(random_sphere(dim_x, dim_y, self.eps, np.inf), shape=tf.shape(x_batch))
		delta = tf.cast(delta, tf.float32)
		delta_grad = self.loss_gradient_delta_framework(x_batch, y_batch, delta)
		#delta = tf.clip_by_value(delta + (self.eps * tf.math.sign(delta_grad)), -self.eps, +self.eps)
		delta = tf.clip_by_value(delta + 1.25 * self.eps * tf.math.sign(delta_grad), -self.eps, +self.eps)
		# Compute and return perturbation
		x_adv = tf.clip_by_value(x_batch + delta, 0, 1)
		return x_adv

	# Override of the train_step implementing Fast better than Free logic
	def train_step(self, data):
		# Unpack data: can be (train, test) or dataset
		# For now assume data is given as the former
		x, y = data

		# Compute dimensions for the l_inf ball		
		batch_size = tf.shape(x)[0]
		dim_flat_channels = np.prod(x.shape[1:])
		
		# Compute the FGSM Example
		x_adv = tf.cond(tf.equal(batch_size, 128), 
			lambda: self.fsgm_rand_delta(128, dim_flat_channels, x, y),
			# Different batchsize for the last batch
			lambda: self.fsgm_rand_delta(50000%128, dim_flat_channels, x, y)
		)
		
		with tf.GradientTape() as tape:
			y_pred = self(x_adv, training = True) # Fw pass
			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update model's weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(y, y_pred)
		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}