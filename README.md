# Robustness of deep neural networks with KAFs trainable activation functions
### *Can activation functions play a role in making neural networks more robust?*

---

This repository contains code and (adversarially) trained models to reproduce results covered in ~~```thesis.pdf```~~, my final project for the master course in Computer Science @ Sapienza University of Rome

## Introduction

- [ ] What the problem is 
- [ ] How we thought we could takle the problem
- [ ] What did we achieve

## Foundamentals

- [ ] Adversarial Attacks theory
- [ ] Different proposals for defenses
- [ ] Kernel Based Activation Functions

## Related Works

- [ ] K-Winners Take All paper
- [ ] Smooth Adversarial Training paper

## Solution Approach

- [ ] CIFAR-10
- [ ] Comparing the activations's distributions for different activation functions (ReLU, KWTA, Kafs) seem to suggest Kafs might be good candidates to improve model robustness (why is that so? We did not investigate, only some empirical evidence)
- [ ] On the limitations of current Lipischitz-Constant based approaches especially when involving Kafs
- [ ] Adversarial training (Madry et al.) and current methods to improve the efficiency (Fast is better than free)

## Evaluation

- [ ] VGG inspired architectures results
- [ ] The exploding gradients problem with KafResNet, why is it happening? (still to clarify)
- [ ] ResNet20 results

## Future Works

- [ ] Different Kernels, resolve the exploding gradient problem and scale to ImageNet
- [ ] Perform more adaptive attacks to assess the robustness of kafresnets as is the current standard (Carlini et al.)

## Conclusions

- [ ] This thesis tries to add to the bag of evidences in literature that smoother architectures might benefit improvements in adversarial resiliency