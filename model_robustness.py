import tensorflow as tf
from tensorflow_addons.optimizers import TriangularCyclicalLearningRate
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import zipfile

# Attack
from art.classifiers import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent

# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configure GPU
# Hardware specific, comment out or tweak as your own preferences
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_visible_devices(gpus[5],'GPU')

print('\n Loading CIFAR10...\n')
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train/ 255.0, x_test/ 255.0

model_path = str(sys.argv[1])

# If model given as a zip file
if zipfile.is_zipfile(model_path):
  f = zipfile.ZipFile(model_path)
  f.extractall() # Extract the archive inside the working directory
  model_path = os.getcwd()+'/'+f.namelist()[0]

print('\nLoading model...\n')
model = tf.keras.models.load_model(model_path)
print('\n..Done\n')

# Wrap the model in a TensorFlowV2Classifier object
model_art = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(32, 32, 3), loss_object=tf.keras.losses.SparseCategoricalCrossentropy(), clip_values=(0, 1))

# Set up a PGD attack w/ parameters as described in FBF ref paper
pgd = ProjectedGradientDescent(estimator= model_art, eps= 8/255, eps_step=2/255, max_iter=50, num_random_init=10, norm=np.inf)

# Create PGD Examples
print('\nCrafting pgd adversarial examples for the whole test set, this might take a while...\n')
adv_examples = pgd.generate(x_test)
print('\n..Done\n')

def find_original_img(perturbed_img, set='test'):
    """Retrieve original image id given an adversarial example""" 
    min_diff = np.math.inf
    index = 0
    
    if set == 'test':
      for i in range(0, x_test.shape[0]):
          diff = np.sum(abs(perturbed_img - x_test[i]))
          if diff < min_diff:
              min_diff = diff
              index = i
    
    elif set =='train':
      for i in range(0, x_train.shape[0]):
        diff = np.sum(abs(perturbed_img - x_train[i]))
        if diff < min_diff:
            min_diff = diff
            index = i
    else:
      print('Arg set must be either train or test')

    return index

# CIFAR10 Classes
classes = ['airplane', 
           'automobile', 
            'bird', 
            'cat', 
            'deer', 
            'dog', 
            'frog', 
            'horse', 
            'ship', 
            'truck']

def attack_succeed(original_img, idx, perturbed_img, model, plot = False, set='test'):
    """
        Return 0 if the model has been fooled 1 otherwise. Plot True if you want more info on the error margin
    """

    if plot:
        # Plot original img alongside with the perturbed one
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(original_img);
        ax[0].title.set_text('Original')
        ax[0].axis('off')
        ax[1].imshow(perturbed_img)
        ax[1].title.set_text('Perturbed')
        ax[1].axis('off')
        plt.show()
        plt.close()

    # Predict
    perturbed_img = tf.expand_dims(perturbed_img, 0)

    if set=='test':
      original = classes[y_test[idx][0]]
    elif set =='train':
      original = classes[y_train[idx][0]]
    else:
      print('Arg set must be either train or test')

    perturbed = classes[np.argmax(model.predict(perturbed_img))]

    if plot:
        print("Real class: {}".format(original))
        print("Predicted class: {} with {} confidence".format(perturbed , round(np.max(model.predict(perturbed_img)) * 100)) )

    return (original == perturbed)


print('\nEvaluating the Robustness of the network...\n')
nfool = 0
nacc = 0
for perturbed_img in adv_examples:
    
    original_idx = find_original_img(perturbed_img)
    nacc += attack_succeed(x_test[original_idx], original_idx, x_test[original_idx], model_art, plot=False) 
    nfool += attack_succeed(x_test[original_idx], original_idx, perturbed_img, model_art, plot=False) 

print("\nAccuracy on clean examples {}%\nAccuracy on adv. examples{}%".format(nacc/10000, nfool/10000))

