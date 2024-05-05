---
layout: post
title: "Optimizing Linear Regression with Gradient Descent"
date: 2023-03-26 17:50:20 +0100
categories: ml
---

In the realm of machine learning, linear regression serves as a fundamental building block for predictive modeling.
However, achieving optimal performance often entails fine-tuning model parameters to minimize errors. 
This article delves into the concept of optimizing linear regression using gradient descent, a powerful optimization algorithm.

By iteratively adjusting model coefficients based on the gradient of the loss function, gradient descent facilitates the convergence towards the optimal solution. 
The article explores the intuition behind gradient descent, its implementation in linear regression, and strategies for enhancing convergence speed and stability.

# Linear Regression

During the process of Supervised Learning, we devise a hypothesis $$h$$ that is a function that will attempt to predict a new outcome given the input features to it.
A standard workflow in Supervised Learning scenarios is that we train some learning algorithm on some training data, and once trained, we try our best to use that algorithm to predict outcomes based on new inputs.

For example, take the problem of trying to predict the price of a house based on the area in square feet. This problem takes one feature into consideration,
the area, which we will call $$x$$, and wants to find the price, which we can call $$h(x)$$.

We can describe the following problem’s hypothesis as the following:

$$
h(x) = \theta_{0} + \theta_{1}x_{1}
$$

The variables for theta refer to the **parameters** of the learning algorithm.
The job of the learning algorithm is to choose the best possible values for theta to give the best approximation for our function $$h(x)$$.

We can also generalize this equation to apply to learning algorithms that take more than one feature into consideration:

$$
\displaylines{
h(x) = \sum_{j=0}^{n}\theta_{j}x_{j} \\\
\text{where } x_{0}=1 \text{ and } n=\text{# of features}
}
$$

To summarize, the function above, $$h(x)$$, is called the hypothesis, and its primary function is to estimate the output for any given set of features $$x$$ based on parameters theta.

Well, how do we know if the result of the hypothesis is accurate? To do this, we utilize a **Cost Function**. The Cost Function aims to predict how accurate the hypothesis is to the actual results.

A Cost Function commonly utilized in Linear Regression for ML models is as follows:

$$
\displaylines{
J(\theta) = \frac{1}{2}\sum_{i=1}^{m}(h(x^{(i)})-y^{(i)})^{2} \\\
\text{where }m = \text{# of samples in training data}
}
$$

Essentially, this function calculates the sum of the squared differences between the predicted value and the actual value from the data contained within our training set.
This is why its also sometimes referred to as the “squared error”.

# Why squared error?

Representing the cost function as a squared difference allows the shape of the surface created by our cost function with respect to our parameter vectors $$\theta$$ to be a parabolic shape,
thus containing no other local minimum other than the global minimum. This allows us to have only one point of convergence for our Gradient Descent algorithm for Linear Regression to approach.
If we had used a quartic function, we could not guarantee that the minima we are approaching would be the global minima, meaning that we could accidentally be updating the function to move closer to an sub-optimal value.

To summarize, the mathematical properties of the squared error allow the resulting surface of the Cost Function to be convex, guaranteeing that every Gradient Descent operation approaches one and only one global minimum.

We also multiply this sum by one-half to simplify the differentiation of this function that will occur in the next step.
Logically, this scalar multiplication does not affect the result of our Cost Function, as we could always multiply it by its reciprocal to undo this action.

Our goal when training ML models using Linear Regression is to minimize this cost function, $$J(\theta)$$, in order to be able to more finely tune our model to be able to output the best possible result.
To minimize this cost function, we apply a technique called **Gradient Descent**.

# Gradient Descent

Before we dive into how Gradient Descent is utilized to help tune our model, let's first explain how we would initialize theta, as it is an important variable in our functions.
Remember, each theta vector $$\theta_{j}$$ represents the values of the parameters of our function that correspond with a feature of our model $$x_{j}$$.
Since our algorithm will be adjusting itself on the fly, it does not matter what theta is initialized as, since it will be updated on every operation made by Gradient Descent.
In practice, most models would initialize it as a vector with size $$n+1$$ of the value $$0$$ in each position where $$n$$ is the number of features of the input, or just a random set of numbers within a specified range.

For our examples, we shall initialize theta as $$\theta = \overrightarrow{0}$$. This notation means a vector of all zeroes.
In the following material, the $$:=$$ symbol will be used to denote assignment whereas the $$=$$ symbol will be used to denote equality.

The Gradient Descent algorithm is defined as follows:

$$
\theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}}J(\theta)
$$

We can also expand the function by substituting in the definition of the cost function:

$$
\theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} [\frac{1}{2}\sum_{i=1}^{m}(h(x^{(i)})-y^{(i)})^{2} ]
$$

If we were to apply one iteration of Gradient Descent and simplify it by taking the partial derivative of the cost function with respect to $$\theta_{j}$$
(again, while expanding $$h(x)$$), we would be updating $$\theta_{j}$$ like so:

$$
\theta_{j} := \theta_{j}-\alpha \sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})x_{j}^{(i)}
$$

This function will be repeated for each value of $$j$$, where $$j=0,1..n$$. The variable, $$\alpha$$, is called the learning rate.
This is a value that is preassigned, typically with a value of $$0.01$$ if the features are scaled to a value of $$\pm 1$$, that will help guide the cost function in incremental steps towards the global minimum.
Think of it as the step value of the function. The larger $$\alpha$$ is, the further the function will travel down the curve per operation.
Therefore, in practical environments, it is best to play around with values of $$\alpha$$ to find a balance to avoid over-correcting or under-correcting the function when performing Gradient Descent.


Looking at the function defined above, it would seem that applying Gradient Descent would be quite straightforward. However, imagine that we have a large training dataset with $$m$$ in the range of hundreds of millions.
The dataset could be hundreds or thousands of terabytes! Currently how our function is defined, we would have to, just for a single update of $$\theta_{j}$$, calculate the partial derivative of the cost function for
every $$i$$ and add those all up. This would be incredibly computationally expensive, and would only get us a single step closer to the global minimum.
Because of this, this version of the Gradient Descent algorithm that we have defined above is referred to as **Batch Gradient Descent**, as it computes each operation of Gradient Descent with the whole batch of samples
from our training dataset.

So what are some alternatives? Thankfully, we have **Stochastic Gradient Descent**. This version of the Gradient Descent algorithm is better adapted to work with larger datasets as it does not require all the training
sample data to be computed at each step. The algorithm is written as follows:

$$
\displaylines{
\text{for i=1..m do } \\\
\theta_{j} := \theta_{j} - \alpha (h(x^{(i)})-y^{(i)})x_{j}^{(i)}
}
$$

Instead of calculating the sum of the partial derivative of each sample within the dataset, we instead update $$\theta_{j}$$ at every sample $$i$$. This means that we will never converge onto the actual global optimum,
as it is a rough approximation at each step, causing our function to oscillate towards the global optimum.
However, in most cases, this implementation provides a sufficient estimation while being much more efficient compared to the Batch Gradient Descent algorithm when working on large datasets.
This is why Stochastic Gradient Descent is more prevalent in real machine learning models in the industry.

Another optimization tactic that is used to narrow the steps of the Stochastic Gradient Descent function as it approaches the global minimum is to decrease the learning rate, $$\alpha$$.
As we discussed earlier, the learning rate is a scalar factor that determines how far $$\theta_{j}$$ should move after each iteration.

# The Normal Equation

What if I told you there is a way to skip this iterative process of calculating the optimal parameters for $$\theta$$?

We can do this by utilizing **the Normal Equation** for Linear Regression. This is a method of matrix multiplication that allows you to skip that whole iterative process of Stochastic Gradient Descent for Linear Regression and
jumps straight to the global optimum (albeit with quite a few matrix operations). We can represent our Gradient Descent function as a Matrix Multiplication problem to allow us to be able to implement it later.

For the following equations, we will use the $$\triangledown$$ symbol to denote a differentiation operation.

We can redefine the derivative of the Cost Function like so:

$$
\triangledown_{\theta}J(\theta) = \triangledown_{\theta}\frac{1}{2}(X\theta - y)^{T}(X\theta - y) \\
$$

$$
\displaylines{
\text{where } X=\begin{bmatrix}
... & (x^{1})^{T} & ...\\\
... & ... & ... \\\
... & (x^{m})^{T} & ... \\\
\end{bmatrix},
\theta = \begin{bmatrix}
\theta_{0} \\\
... \\\
\theta_{n}
\end{bmatrix},
y = \begin{bmatrix} 
y^{(1)} \\\
... \\\
y^{(m)}
\end{bmatrix} \\\
n = \text{# of features} \\\
m = \text{# of samples in training data}
}
$$

The matrix $$X$$ is called the **design matrix**, and is essentially all of the feature vectors, $$x$$, transposed from column vectors into row vectors.
The vector $$y$$ is a column vector that represents the labels of your training samples.
As you can see, the matrix product of $$X$$ and $$\theta$$ results in a column vector which represent the results of the hypothesis, $$h(x)$$.
Subtracting that product from $$y$$ will then give you the error of the model, thus, we can see that we have translated our previous equation for the cost function into a matrix multiplication problem.

If we differentiate the cost function now, we will arrive at:

$$
J(\theta) = X^{T}X\theta - X^{T}y
$$

We can set this equation to $$\overrightarrow{0}$$ to represent our starting values for our parameters. Solving for $$\theta$$, we get:

$$
\displaylines{
X^{T}X\theta - X^{T}y = \overrightarrow{0} \\\
\theta = (X^{T}X)^{-1}X^{T}y
}
$$

Plugging in the corresponding values for the matrices will allow you to, in one step, arrive at the optimal values for $$\theta$$. Although it would seem pretty straightfoward
to carry out these matrix operations, we have to take into consideration the computational complexity associated with this method.

Matrix multiplication is known to be $$O(n^{3})$$ for square matrices. This is because computing each position of the resulting matrix would require a dot product,
which in itself takes $$O(n)$$ time as it is a linear combination of the products of each element at a given position. Since the matrices are squares, you would need to perform
the dot product operation $$n^{2}$$ times, where $$n$$ is the number of elements in each matrix.

Therefore, it would be more computationally expensive to use the Normal Equation for Linear Regression as opposed to using Stochastic Gradient Descent.

{% include mathjax.html %}
