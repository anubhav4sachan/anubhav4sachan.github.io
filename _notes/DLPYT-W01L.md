---
layout: note
title: History, motivation, and evolution of Deep Learning
lnumber: 01
uni: New York University
number: dlpyt
ctitle: Deep Learning (with Pytorch)
version: Spring 2020
ytlink: https://www.youtube.com/watch?v=0bMe_vCZo30&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq
website: https://atcold.github.io/pytorch-Deep-Learning/
instructor: Yann LeCun
ilink: http://yann.lecun.com/
---
<hr>

## Part A
### Section 1
- Backpropagation (1980s) requires an activation function which should be continuous (not discrete, as thought till 80s). Since we didn't have correct neurons, hence field died eventually and reemerged in 80s through backpropagation
- 1995 - 2010 died again, reemerged with speech recognition and used deep neural networks
- 2012 - Imagenet
- 2016 - Again a boom, NLP

##### Supervised Learning
- 90% application
- parameterized function to tweak
- gradient computation and backprop find how to tweak

### Section 2
- MP models started feature engineering and use it in trainable classifier (computes weighted sum and threshold)
- DL -> cascade *non-linear* (because if linear, two linear can collapse into 1 linear and has no use) modules for feature engineering and classification *end to end*
- (Deep) Multi Layer Neural Net -> each unit (computes weighted sum) is passed through a non-linear $$f(x)$$ (the activation $$f(x)$$) and the weights are learned through *learning algorithms*
- **Supervised Learning** -> Function Optimization
- loss module calculates divergence/discrepancy/distance with the target and helps learning the params (weights)
- How? Gradient Descent. (Hill example: person wants to go to the village at (*convex*) valley from hilltop)
- Objective $$f(x)$$ -> avg over millions of samples and can't be done taking all the examples at once, and hence use *mini-batches*
- *mini-batches* will make training erratically but generalized.
-  Noise helps generalize

### Section 3
- Computation of Gradients by Backprop using chain rule
- not practical to use whole 256x256 column vector, hence, compress like in eyes (from 100M to 1M)
- feedforward must be fast
- Wiesel -> simple cells detect local features and complex cells *pool* the outputs of simple cells. 
	- Even if (due to different orientation) different simple cells are activated, the *pooled* complex cells activate same set of neurons
- above point serves as an inspiration for CNN
- Earlier, segment and then get the object, but with CNN, it's now done end to end.

## Part B
### Section 1
- **DL Revolution**
	- Deep ConvNets for Object Recognition (on GPU)
	- *Joke:* You can't get a paper published in CV Conf, without Neural Nets!

- **Multi-layer Architectures**
	- Find Compositional Structure -> learn something in each layer.
	- low level -> mid level -> high level -> trainable classifier
	- One pass object detection (retina net)
	- Masked CNN
	- Panoptic instance Segmentation
	- MRI 3D ConvNet

### Section 2
- **Why does it work so well?**
	- We can approximate any function with two layers
		- Why do we need layers?
	- What is so special convolutional networks?
		- Why do they work so well on natural signals?
	- The objective function are highly non-convex.
		- Why doesn’t SGD get trapped in local minima?
	- The networks are widely over-parameterized.
		- Why do they not overfit?

- **DL -> Representation Learning**
	- *Basic:* expanding the dimension of the representation so that things are more likely to become linearly separable.
	- Random Projection: First layer -> projections (i.e. Random weights)
	- Extreme NN (completely stupid): Two layered NN with one trainable layer, and other fixed with random weights (usually, first one)

### Section 3
- **Hierarchical Representations**
	- Image:
		- Pixel → edge → texton → motif → part → object
	- Text:
		- Character → word → word group → clause → sentence → story
	- Speech:
		- Sample → spectral band → sound → ... → phone → phoneme → word

- **Do we really need deep architectures?**
	- SVM -> 2 Layered NN with stupid 1st layer initialization and 2nd is just an fc.
	- DL -> More efficient in learning *representations*
		- Why?
			- $$y = F(W^k \cdot F(W^{k-1} \cdot F(... \cdot F(W^0 \cdot X)...))) $$
			- More Complex  $$f(x)$$ with less hardware
			- Trades space for time
				- More layers -> more sequential computation
				- less hardware -> less parallel computation