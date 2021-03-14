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
	- Computing $$G$$ shall involve complicated algorithms
	- The cost function $$C(y, \bar{y})$$ computes the similarity (or difference) between the prediction and true label. 
	- Example:
		- Linear Model: $$\bar{y} = \sum_{i} w_i \cdot x_i$$
		- Nearest Neighbor: $$\bar{y} = \arg \min_k \Vert x - w_k\Vert ^2$$

- Loss function -> minimized during training 
	- per-sample loss \
		$$L(x, y, w) = C(y, G(x, w))$$
	- A set of samples $$S$$ \
		$$S = {(x[p], y[p]), \ \text{where} \ p = 0 ... P-1}$$
	- Average Loss ($$\mathcal{L}(S, w)$$): loss over all the samples -> Actually minimized during the training

	   $$\mathcal{L}(S, w) = \frac{1}{P} \sum_{p = 0}^{P-1} L((x[p], y[p]), w)$$
-  ML is basically all about optimizing the functions: Minimizing (usual), Maximizing (reward functions in RL)

-  Gradient based methods: methods finding minima/maxima using gradients (function must differentiable over the range)
	-  Full (batch) Gradient: $$w \leftarrow w - \eta \frac{\partial \mathcal{L}(S, w)}{\partial w}$$
		-  If $$\eta$$ is scalar const, hence step is taken in direction given by gradient (always takes steepest (orthogonal) step). However, if $$\eta$$ is a matrix (_second order methods_), then move in a direction pointed by the vector in the direction of gradient (not necessarily steepest).

-  For non-differentiable functions, _zero-th order methods_ or _gradient free methods_ are used.

-  Deep Learning is all about on Gradient based Methods, RL does not have differentiable cost function (usually). $$C(S, w)$$ gives you the rewards, and since $$y$$ is unknown, you cannot calculate the gradient to find out if the direction chosen by the agent was correct or not. Hence, reiterate from starting -> modify $$\bar{y}$$, and observe how $$C(S, w)$$ behaves, and work efficiently to maximize $$C(S, w)$$.  It can be inferred, as the space of $$\bar{y}$$ increases, the complexity of RL increases.
  
### Section 2
- Stochastic Gradient (SGD):
	- Instead of full gradient (over all samples), use gradient descent for a single example as:

		$$w \leftarrow w - \eta \frac{\partial L((x[p], y[p]), w)}{\partial w}$$

  - Since, doing SGD on single example will give a noisy trajectory for reaching minima, hence, people do it in batches for parallelization (thereby improving efficiency of computation).

- Traditional Neural Net:
	- Stacked linear and non-linear functional blocks
	- Basically, a lot of matrix products and summation, hence processing in GPUs is faster than CPUs [[reason]](https://ai.stackexchange.com/questions/21938/how-do-gpus-faciliate-the-training-of-a-deep-learning-architecture).
	- Backpropagation through the non-linear function, to compute $$\frac{\text{d}C}{\text{d}x}$$
	- 

### Section 3
- For a neural net with forward pass:

	$$x \rightarrow f(x, w_f) \rightarrow z_f \rightarrow g(z_f, w_g) \rightarrow z_g \rightarrow C(z_g, y) \rightarrow C$$

	- Chain rule for vector vector functions will be

		$$z_g:[d_g \times 1], \ z_f:[d_f \times 1]$$

		$$\frac{\partial C}{\partial z_f} = \frac{\partial C}{\partial z_g} \frac{\partial z_g}{\partial z_f}$$

		$$[1 \times d_f] = [1 \times d_g] \text{*} [d_g \times d_f]$$
		
	- What is $$\frac{\partial z_g}{\partial z_f}$$?

		Ans: Jacobian Matrix ([basic idea](https://math.stackexchange.com/questions/14952/what-is-the-jacobian-matrix/1127350#1127350)) ([detailed](https://math.stackexchange.com/questions/14952/what-is-the-jacobian-matrix/14955#14955))

		Partial Derivative of _i_-th output w.r.t _j_-th output. If you twiddle with _j_-th input, it would affect the whole output matrix.
		
		$$\Bigg(\frac{\partial z_g}{\partial z_f}\Bigg)_{ij} = \frac{(\partial z_g)_{i}}{(\partial z_f)_j}$$
		
		
## Part B
### Section 1
- Basic Modules
	- _Linear_

		$$Y = W.X; \ \ \ \frac{\text{d}C}{\text{d}X} = W^\top \frac{\text{d}C}{\text{d}Y}$$

	- _ReLU_

		$$y = \text{ReLU}(x); \ \ \ \frac{\text{d}C}{\text{d}X} = 
		\left\{
		\begin{array}{ll}
		0, & \text{if } x < 0\\
		\frac{\text{d}C}{\text{d}Y} & \text{otherwise}
		\end{array}
		\right.$$

	- _Duplicate_

		$$Y_1 = X, Y_2 = X; \ \ \ \frac{\text{d}C}{\text{d}X} = \frac{\text{d}C}{\text{d}Y_1} + \frac{\text{d}C}{\text{d}Y_2}$$
	
		Note that, the gradients are summed. Although, PyTorch does it for me, but take note.
		
	- _Add_

		$$Y = X_1 + X_2; \ \ \ \frac{\text{d}C}{\text{d}X_1} = \frac{\text{d}C}{\text{d}Y}, \frac{\text{d}C}{\text{d}X_2} = \frac{\text{d}C}{\text{d}Y}$$
		
	- _Max_
		
		$$y = \max({x_1, x_2}); \ \ \ \frac{\text{d}C}{\text{d}x_1} = 
		\left\{
		\begin{array}{ll}
		\frac{\text{d}C}{\text{d}y}, & \text{if } x_1 > x_2\\
		0 & \text{otherwise}
		\end{array}
		\right. $$
		
	- _LogSoftmax_ 
		
		$$Y_i = X_i - \log \left[\sum_{j} \exp (X_j) \right]$$