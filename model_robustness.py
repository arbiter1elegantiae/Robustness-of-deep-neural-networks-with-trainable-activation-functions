# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
# Main imports
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import sys

import activationsf


# %%
# Server config
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_visible_devices(gpus[5],'GPU')


# %%
# Load CIFAR10
print('Loading CIFAR10...')
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train/ 255.0, x_test/ 255.0


# %%
# Load robust model trained as https://arxiv.org/abs/2001.03994 which you want to test robustness
model_path = str(sys.argv[1])
model = tf.keras.models.load_model(model_path)


# %%
from art.classifiers import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent

# Wrap the model in a TensorFlowV2Classifier object
model_art = TensorFlowV2Classifier(model=model, nb_classes=10, input_shape=(32, 32, 3), loss_object=tf.keras.losses.SparseCategoricalCrossentropy(), clip_values=(0, 1))


# %%
# Set up a PGD attack w/ parameters as described in FBF ref paper
pgd = ProjectedGradientDescent(estimator= model_art, eps= 8/255, eps_step=2/255, max_iter=50, num_random_init=10, norm=np.inf)



# %%
# Create n adversarial PGD Examples
print('Crafting pgd adversarial examples for the whole test set, this might take a while...')

adv_examples = pgd.generate(x_test)

print('Done, storing them...')


# %%
# If needed store them
model_name = model_path.rsplit('/', 1)[-1]
np.save('./'+model_name+'_adv_examples.npy', adv_examples)



# %%
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


# %%
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


# %%
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


# %%
# Robustness Accuracy, set plot=True if you want image-wise stats
# last line, /|adv_examples|
print('Evaluating the Robustness of the network...')
nfool = 0
nacc = 0
for perturbed_img in adv_examples:
    
    original_idx = find_original_img(perturbed_img)
    nacc += attack_succeed(x_test[original_idx], original_idx, x_test[original_idx], model_art, plot=False) 
    nfool += attack_succeed(x_test[original_idx], original_idx, perturbed_img, model_art, plot=False) 

print("Accuracy on clean examples {}%\nAccuracy on adv. examples{}%".format(nacc/10000, nfool/10000))


