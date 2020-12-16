---
layout: note
title: Gradient Descent and Backpropagation Algorithm
lnumber: 02
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

### Some points from W01: Practicum

**Link:** [Prof. Canziani's ipynb](https://github.com/Atcold/pytorch-Deep-Learning/blob/master/02-space_stretching.ipynb)

- RGB: R = X-Axis, G = Y-Axis
- -ve determinant denotes reflection
-  Linear transforms (`nn.Linear()`) can rotate, reflect, stretch and compress, but cannot curve, hence, non-linearity is required.

<hr>

**W02L: Gradient Descent and Backpropagation Algorithm**

## Part A
### Section 1
- Parameterize Model: $$\bar{y} = G (x, w)$$
	- Computing G shall involve complicated algorithms
	- The cost function C(y, \bar{y}) computes the similarity (or difference) between the prediction and true label. 
	- Example:
		- Linear Model:- \bar{y} = \sum_i w_i \cdot x_i
		- Nearest Neighbor:- $$\bar{y} = \arg \min_k \Vert x - w_k, . \Vert ^2$$
- Loss function -> minimized during training 
	- per-sample loss
		- L(x, y, w) = C(y, G(x, w))
		- A set of samples S
			- S = {(x[p], y[p]), where p = 0 ... P-1}
		- Average Loss (\mathcal{L}(S, w)): loss over all the samples -> Actually minimized during the training
		- $$\mathcal{L}(S, w) = \frac{1}{P} \sum_{p = 0}^{P-1} L((x[p], y[p]), w)$$
-  ML is basically all about optimizing the functions: Minimizing (usual), Maximizing (reward functions in RL), 






