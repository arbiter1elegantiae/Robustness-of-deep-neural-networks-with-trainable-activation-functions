# Sanity check for the latter function using kaf_cnn model
# Check that its prediction matches an adhoc model with weights loaded from the desired model
# Test result: WORKS
tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

kaf1_inp = get_hidden_layer_input(kaf_cnn, 'kaf_1', kaf_test_sample)

kaf_cnn_kaf1 = Sequential([
    layers.Conv2D(32, 3, padding='same', activation=None, kernel_initializer='he_uniform',  input_shape = (32, 32, 3), name = 'new_conv'),
    layers.BatchNormalization(name = 'new_bn') ])

# Set weights from kaf_cnn
wgts_conv = kaf_cnn.layers[0].get_weights()
kaf_cnn_kaf1.get_layer('new_conv').set_weights(wgts_conv)

wgts_bn = kaf_cnn.layers[1].get_weights()
kaf_cnn_kaf1.get_layer('new_bn').set_weights(wgts_bn)

do_match = tf.reduce_sum(tf.cast(tf.math.logical_not(tf.equal(kaf1_inp, kaf_cnn_kaf1.predict(kaf_test_sample))), dtype=tf.float32))

print (do_match) # 0.0 for True, False otherwise






import logging
import time

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.defences.trainer.adversarial_trainer_FBF import AdversarialTrainerFBF
from art.utils import random_sphere

logger = logging.getLogger(__name__)

class AdversarialTrainerFBFTensorflow2(AdversarialTrainerFBF):
	"""
	Class performing adversarial training following Fast is Better Than Free protocol.
	| Paper link: https://openreview.net/forum?id=BJx040EFvH
	| The effectiveness of this protoocl is found to be sensitive to the use of techniques like
	data augmentation, gradient clipping and learning rate schedules. Optionally, the use of
	mixed precision arithmetic operation via apex library can significantly reduce the training
	time making this one of the fastest adversarial training protocol.
	"""
	
	def __init__(self, classifier, eps=8/255, use_amp=False, **kwargs):
		"""
		Create an :class:`.AdversarialTrainerFBFTensorflow2` instance.
		:param classifier: Model to train adversarially.
		:type classifier: :class:`.Classifier`
		:param eps: Maximum perturbation that the attacker can introduce.
		:type eps: `float`
		:param use_amp: Boolean that decides if apex should be used for mixed precision arithmantic during training
		:type use_amp: `bool`
		"""
		super().__init__(classifier, eps, **kwargs)
		self._use_amp = use_amp

	
	def loss_gradient_delta_framework(self, x, y, delta, **kwargs):
		"""
		Compute the gradient of the loss function w.r.t. `delta`.
		:param x: Input with shape as expected by the model. 
		:param y: Indices of shape (nb_samples,).
		:param delta: Input with shape as expected from the model
		:return: Gradients of the same shape as `x`.
		"""

		if self._classifier._loss_object is None:
			raise ValueError("Loss object is necessary for computing the loss gradient.")


		x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
		delta_tf = tf.convert_to_tensor(delta, dtype=tf.float64)
		
		if tf.executing_eagerly():
			with tf.GradientTape() as tape:
				#tape.watch(x_tf)
				tape.watch(delta_tf)
				model_outputs = self._classifier._model(x_tf + delta_tf)

				loss = self._classifier._loss_object(y, model_outputs)
				loss_grads = tape.gradient(loss, delta_tf)

		else:
			raise NotImplementedError("Expecting eager execution.")

		return loss_grads


	def fit(self, x, y, validation_data=None, batch_size=128, nb_epochs=20, **kwargs):
		"""
		Train a model adversarially with FBF protocol.

		:param x: Training set.
		:type x: `np.ndarray or tf.tensor` 
		:param y: Labels for the training set.
		:type y: `np.ndarray or tf.tensor`
		:param validation_data: Tuple consisting of validation data
		:type validation_data: `np.ndarray or tf.tensor`
		:param batch_size: Size of batches.
		:type batch_size: `int`
		:param nb_epochs: Number of epochs to use for trainings.
		:type nb_epochs: `int`
		:param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of the target classifier.
		:type kwargs: `dict`
		:return: `None`
		"""
		n_batches = int(np.ceil(len(x) / batch_size))
		ind = np.arange(len(x))

		# Implementation of cyclic lr to improve training time (see ref paper sec4.2)
		def lr_schedule(t):
			return np.interp([t], [0, nb_epochs * 2 // 5, nb_epochs], [0, 0.21, 0])[0]


		def _batch_process(x_batch, y_batch, lr, opt):
			"""
			Perform the operations of FBF for a batch of data.
			:param x_batch: batch of x.
			:type x_batch: `np.ndarray`
			:param y_batch: batch of y.
			:type y_batch: `np.ndarray`
			:param lr: learning rate for the optimisation step.
			:type lr: `float`
			:return: `(float, float, float)`
			"""
			n = x_batch.shape[0]
			m = np.prod(x_batch.shape[1:])
			# Set a random init for delta w/ L_inf = 8/255 and compute gradients wrt x+delta
			delta = random_sphere(n, m, self._eps, np.inf).reshape(x_batch.shape).astype(ART_NUMPY_DTYPE)
			delta_grad = self.loss_gradient_delta_framework(x_batch, y_batch, delta)
			delta = np.clip(delta + 1.25 * self._eps * np.sign(delta_grad), -self._eps, +self._eps)
			x_batch_pert = np.clip(x_batch + delta, self._classifier.clip_values[0], self._classifier.clip_values[1])
			
			
			# Predict, loss and apply a SGD step
			with tf.GradientTape() as tape:
				model_outputs = self._classifier._model(x_batch_pert)
				loss = self._classifier._loss_object(y_batch, model_outputs)

			g = tape.gradient(loss, self._classifier._model.trainable_variables)

			opt.apply_gradients(zip(g, self._classifier._model.trainable_variables))
			
			train_acc = ((np.expand_dims(np.argmax(model_outputs, axis = 1), -1) == y_batch).sum()/ n) * 100

			return loss, train_acc

		# Training
		for i_epoch in range(nb_epochs):

			#_a = [v for v in self._classifier._model.trainable_variables if "kaf_4/mix_coeffs" in v.name][0][0][0][0][0]
			#print(_a)
			## Get Kaf's invariants: kernel bandwidth and dictionary
			#kaf1 = self._classifier._model.get_layer(name = 'kaf_3')
			#kb = kaf1.k_bandw
			#d_tmp = tf.squeeze(kaf1.d)
			#d = tf.expand_dims(d_tmp, 0)
#
			## We want to evaluate Kafs on the same input: use dictionary itself as activation
			#act = tf.expand_dims(d_tmp, -1)
#
			## Prepare plot settings
			#fig=plt.figure(figsize=(15, 8))
			#plt.subplots_adjust(wspace = 0.5, hspace = 0.3)
			#columns = int(5/2) + 1
			#rows = 2
			#ax = []
		#
			#for i in range(1, 6): # For each Kaf layer
		  #
		  	#	name = 'kaf_'+str(i)
		  	#	layer = self._classifier._model.get_layer(name = name)
#
		  	#	# Get mixing coefficients and compute Kaf
		  	#	a = tf.cast( tf.expand_dims(tf.squeeze(layer.a)[0], 0), dtype = tf.float16 )
		  	#	kaf = activationsf.kafActivation(act, a, d, kb)
#
		  	#	# Plot
		  	#	ax.append( fig.add_subplot(rows, columns, i) )
		  	#	ax[-1].set_title('{}, Epoch {}'.format(name,i_epoch+1))  
		  	#	plt.plot(d_tmp, kaf, 'r')
			#	
			#plt.show()


			
			# Shuffle the examples
			np.random.shuffle(ind)
			start_time = time.time()
			train_loss = 0
			train_acc = 0
			train_n = 0


			for batch_id in range(n_batches): # For each batch
				lr = lr_schedule(i_epoch + (batch_id + 1) / n_batches)
			#	lr = 0.15
				optsgd = tf.keras.optimizers.SGD(learning_rate=lr)

				# Create batch data
				x_batch = x[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]].copy()
				y_batch = y[ind[batch_id * batch_size : min((batch_id + 1) * batch_size, x.shape[0])]]

				# Actual FGSM learning
				_train_loss, _train_acc = _batch_process(x_batch, y_batch, lr, optsgd)

				train_loss += _train_loss
				train_acc += _train_acc
				train_n += 1

				train_time = time.time()
				# Log epoch statistycs
				if validation_data is not None:
					(x_test, y_test) = validation_data
					output = np.argmax(self.predict(x_test), axis=1)
					nb_correct_pred = np.sum(output == np.argmax(y_test, axis=1))
					logger.info(
						"{} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
						i_epoch,
						train_time - start_time,
						lr,
						train_loss / train_n,
						train_acc / train_n,
						nb_correct_pred / x_test.shape[0],
						)
					)
				else:	
					logger.info(
						"{} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
							i_epoch, train_time - start_time, lr, train_loss / train_n, train_acc / train_n
						)
					)
					print(
						"epoch:{} \t time:{:.1f} \t lr:{:.4f} \t loss:{:.4f} \t acc:{:.4f}".format(
							i_epoch, train_time - start_time, lr, train_loss / train_n, train_acc / train_n
						)
					)

		
	def evaluate(self, x_test, y_test, batch_size=128):

		n_batches = int(np.ceil(len(x) / batch_size))

		n = x_test.shape[0]

		model_outputs = self._classifier._model(x_test)
		
		train_acc = ((np.expand_dims(np.argmax(model_outputs, axis = 1), -1) == y_test).sum()/ n) * 100
		print(train_acc)






	def fit_generator(self, generator, nb_epochs=20, **kwargs):

		print("not yet implemented")

		
