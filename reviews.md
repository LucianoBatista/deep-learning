# Chap 1

Concepts:

- loss function: measure how far this output is from what you expected (objective function or cost function). Predicted - Observed.
- Optimizer: implements backpropagation to minimize the loss function (adjusting the weights).

What is transformative about deep learning is that it allows a model to learn all layers of representation jointly, at the same time, rather than in succession (greedily).

Essential characteristics of how deep learning learns from data: the incremental, layer-by-layer way in wich increasingly complex representations are developed, and the fact that these intermediate incremental representations are learned jointly, each layer being updated to follow both the representational needs of the layer above and the needs of the layer below.

Top **winners kaggle** algos:

1. Keras
2. LightGBM
3. XGBoost
4. PyTorch

Top frameworks used by kagglers:

1. Scikit-learn
2. TensorFlow
3. Keras
4. Xgboost
5. PyTorch
11. Tidymodels

Interesting that tidymodels are very new package and is already on 11 with 7.2% of usage.

Very important characteristic of deep learning:

- **Versatility and reusability**: Unlike many prior machine-learning approaches, *deep-learning models can be trained on additional data without restarting from scratch*, making them viable for continuous online learning—an important property for very large production models. Furthermore, *trained deep-learning models are repurposable and thus reusable*: for instance, *it’s possible to take a deep-learning model trained for image classification and drop it into a video-processing pipeline*. This allows us to reinvest previous work into increasingly complex and powerful models. This also makes deep learning applicable to fairly small datasets.


# Chap 2 (The mathematical building blocks of neural nets)

Working with MNIST dataset.

The core building block of neural networks is the layer. You can think of a layer as a filter for data: some data goes in, and it comes out in a more useful form. 

Layers extract representations out of the data fed into them, hopefully, representations that are more meaningful for the problem at hand. 

Dense layers = fully connected.

- We set two layers

```python
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

- We need 3 more stuffs for out model: **optimizer**, **loss function**, **metrics** to monitor during training and testing.

We have here similar methods like sklearn: 

- .fit: to train
- .predict: to predict
- .evaluate: to test

Concepts:

- tensors: area multidimensional NumPy arrays. Matrices are rank-2-tensors and tensors are a generalization of matrices to an arbritrary number of **dimensions**.
- scalars: rank-0 tensors
- vectors: rank-1 tensors

A tensor is defined by three key attributes:

- number of axes (rank): *ndim*
- shape: how many dimensions the tensor has along each axis.
- data type: as the same as dtype. 

In general, the first axis in all data tensors you'll come across in deep learning will be the sample axis (samples dimension). MNIST samples are images of digits.

Deep learning models don't process an entire dataset at once; rather, they break the data into **small batches**. Also called of batch axis or batch dimension (axis-0).

## Real-world examples of data tensors

- vector data: (samples, features)
- timeseries/sequence data: (samples, timestamps, features)
- images: (samples, height, width, channels)
- videos: (samples, frames, height, width, channels)

## The Gears of neural networks: tensor operations

All transformations learned by deep neural nets can be reduced to a handful of tensor operations (or tensor functions) applied to tensors of numeric data.

### Element-wise operations
### Broadcasting
Operations between different shapes arrays!
### Tensor product
The tensor product, or dot product, is one of the most common, most useful tensor operations.
### Tensor reshaping

## Geometric representation

In general, elementary geometric operations such as translation, rotation, scaling, skewing and so on, *can be expressed as tensor operations*.

- **transaltion**: adding tensors
- **rotation**: via dot product: `R = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]`
- **scaling**: via dot product.
- **linear transform**
- **affine transform**
- **Dense layer with relu activation**

## Important!!

A multi-layer neural net made entirely of dense layers without activations would be equivalent to a single dense layer.

This deep nn would just be a linear model in disguise.

Activations functions **increase the hypothesis spaces** for a dnn.

## The engine of nn: gradient-based optimization

`output = relu(dot(input, W) + b)`

W and b are tensors that are attributes of the layer, called weights (kernel and bias).

The resulting representation on a start point (random init) are meaningless. We need a feedback to gradually learn about the data.

How is made the training loop?

- draw a batch of training samples x and corresponding targets y_true.
- run the model on x (forward pass) to obtain predictions y_pred
- **compute the loss of the model on the batch**, a measure of the **mismatch between y_pred and y_true**.
- update all weights of the model in a way that slightly reduces the loss on this batch.

Gradient Descent: is the optmization technique that powers modern neural nets. 

Being able to derive function is a very powerful tool when it comes to optimization, the task of finding values of x that minimize the value of f(x). 

The derivative completely dewscribes how f(x) evolves as you change x. If you want to reduce the value of f(x), you just need to move x a lttle in the opposite direction from the derivative.

### Derivative of a tensor operation: the gradient

- y = f(x): scalar value x into another scalar y. Possible to look at a curve in a 2D plane
- z = f(x, y): a tuple of scalars into another scalar. Possible to plot as a 2D surface in a 3D plane.
- Likewise, you can imagine functions that take as **input matrices**, functions that take as **input rank-3 tensors**, etc.

As our function is continuous and smooth we can extend the concept of derivative, and call this generalization as **gradient**.

The gradient of a tensor function **represents a curvature of the multidimensional surface** described by the function.

```python
y_pred = dot(W, x) # without bias and activation
loss_value = loss(y_pred, y_true)
loss_value = f(W) # f describes the curve (or high-dimensional surface) formed by loss values when W varies.

```

- w0 = current value of W
- the derivative of f in the point w0 is a tensor grad(loss_value, w0), where each coefficient grad(loss_value, w0)[i, j].
- the tensor grad(loss_value, w0)[i, j] is the gradient of the function f(w) = loss_value in w0. Also called **gradient of loss_value with respect to w around w0**.


### Stochastic gradient descent

Differentiable functions have a minimum, place where the diravative is 0.

mini-batch stochastic gradient descent (mini-batch SGD)
*stochastic == random*

- *learning_rate*: is the parameter that you set for the steps during the gradient descent algo.
- batch gradient descent: uses all the data (very expensive)

Additionaly, there exist multiple variants of SGD that differ by taking into account previous weight updates:
- SGD with momentum
- Adagrad
- RMSprop
- ...

There is a lot of variants and research on optimizers. But, the concept of momentum realy deserves our attention.

Momentum address two issues with SGD: **convergence speed** and **local minima**.

Think the optimization process as a ball rolling down the loss curve.

### Chaining derivatives: the backpropagation algo

Neural Nets consists of many tensor operations chained together. And Calculus tells us that such a chain of functions can be derived using the following identity, called chain rule.


```python
def fg(x):
    x1 = g(x)
    y = f(x1) # f(g(x))
    return y

# chain rule
grad(y, x) == grad(y, x1) * grad(x1, x)
```

## Automatic diff with computation graphs

A computation graph is the data structure at the heart of TF and the deep learning revolution in general.

Computation graph have been an extremely successful abstraction in CS because they enable us to treat computation as data: a computable expression is encoded as a machine-readable data structure that can be used as the input or output of another program.


# Chap 03

To do anything in TF, we're going to need some tensors.

## GradientTape

- new function on tf 2.0
- can be used to write custom training loops
- use automatic differentiation

### Automatic Diff

Set of techniques that can automatically compute the derivative of a function by repeatedly applying the chain rule. The computer program executes a sequence of elementary functions (exp, log, sin, cos, ...).

Can compute **partial derivatives** with respect to many inputs.

When implementing custom training loops with Keras and TF, you do need to define, at a bare minimum, four components:

- **architecture**
- **loss function**
- **optimizer**
- **step function**: that encapsulates the forward and backward pass of the network

## Anatomy of a neural nets

### Layers

- fundamental data structure
- some layers are stateless, but *more frequently layers have a state*: the *layer's weights*, *one or several tensors learned* with stochastic gradient descent, which *together contain the network's knowledge*.
- different types of layers are appropriate for different tensor formats and different types of data processing:
  - simple vector data, **stored in 2D** (samples, features), is often processed by **densely connected layers**.
  - sequence data, **stored in 3D** tensors of shape (samples, timesteps, features), is typically processed bu **recurrent layers, such as LSTM** layer, or 1D convolution layers (Conv1D).
  - Image data, **stored in 4D** tensors, is usally processed by **2D convolution layers (Conv2D)**.

The topology of a model, architecture, represents your *hypothesis space*, your *space of possibilities*. So, is extremely important you're able to choose the best topology/architecture for your problem.

# Chap 04

- classifying movie reviews as positive or negative (binary class)
- classifying news wires by topic (multiclass class)
- estimating the price of a house, given real-state data (scalar regression)

**look at vector regression to detect bounding box of images!!!**

# Chap 05

## Generalization

Refers to how well the trained model performs on data it has never seen before.

## Underfit vs overfit

For every machine learning problem at the beginning of training, optimization and generalization are correlated: the lower the loss on training data, the lower the loss on test data. But, at some point the both curves start to diverge. The training keeps decreasing the loss, while the validation/test start to increase (overfit).

Overfitting is particularly likely to occur when your **data is noisy**, if it **involves uncertainty**, or if it includes **rare features**.

- **Noise data**: Invalid inputs or data was mislabeled.
- **Uncertainty:** Similars class of something: different species of banana, orange or tangerina...
- **rare features:** If you don't see enough data about something you're biased to take wronge conclusions.

It's common to do feature selection before training. The typical way to do is to compute some usefulness score for each feature available: *mutual information between the feature and the labels*.

The nature of generalization in deep learning has rather little to do with deep learning models themselves, and much to do with structure of information in the real world.

### The manifold hypothesis

Look of **MNIST**, 28x28 array and each pixels goes from 0 to 255, that is 256^784 possibilities. Infinite possibilities to represent MNIST numbers, however, very few of these inputs would look like valid MNIST samples (tiny subspace). What's more, this subspace isn't just a set of points sprinkled at random in the parent space: it is highly structured.

**First**, the subspace of valid handwritten digits is continuous.Futher, all samples in the valid subspace are *connected by smooth paths* that run through the subspace. This means that if you take two random MNIST digits A and B, there exists a sequence of "intermediate" images that morph A into B, such that two consecutive digits are very close to each other. 

In technical terms, you would say that *handwritten digits form a manifold within the space of possible 28x28 unit8 arrays*.

*A "manifold" is a lower-dimensional subspace of some parent space, that is locally similar to a linear (Euclidian) space.*

*For instance, as smooth curve in the plane is a 1D manifold within a 2D space, because for every point of the curve, you can draw a tangent*

*A smooth surface within a 3D space is a 2D manifold*

The ability of interpolate between samples is the key to understanding generalization in deep learning.

The manifold hypothesis implies that:

- Machine learning models only have to *fit relatively simple, low-dimensional*, highly-structured subspaces within their potential input space (latent manifolds).
- *Within one of these manifolds*, *it’s always possible to interpolate between two inputs*, that is to say, morph one into another via a continuous path along which all points fall on the manifold.

### Interpolation as a source of generalization

If you work with data points that can be interpolated (some manifold representation), you can start making sense of points you've never seen before, by relating them to other points that lie close on the manifold.

Here we're searching for **Manifold Interpolation** and not Linear Interpolation

Keep in mind that Interpolation can only help you make sense of things that are very close to what you've seen before: it enables *local generalization*.

Humans are capable of extreme generalization, which is enabled by cognitive mechanisms other than interpolation: abstraction, symbolic models, reasoning logic, common sense, innate priors.

By construction, deep learning is about taking a big, complex curve - a manifold - and incrementally adjusting its parameters until it fits some training points.

## Evaluation of the model

- Training/Testing/Validation
- K-Fold validation
- Fancy K-Fold validation
- ...

### Beting a common-sense baseline

Is very important that we have an understanding if our model is good enough. Beside the business requirements your model needs to beat some baseline, may be a simple model, a random classifier... And from that you can estimate if you are doing some meaningful improvement.

### Things to keep in mind about evaluation

- Data representativeness (distribution of classes, means, metrics...)
- The arrow of time (data leakage)
- Redundancy in your data (duplication)


## Improving model fit

- Tuning key gradient descent parameters: learning_rate for example
- Better architecture
- Increasing model capacity: bigger layers, more layers, different kind of layers (more representional power).


## Improving generalization

- dataset curation: better data, less noisy

*"Spending time doing data collection almost always yields a much greater return on investment that spending the same on developing a better model."*

- make sure you have enough data
- minimize labeling erros
- clean your data and deal with missing values
- if you have many features and you aren't sure which ones are actually useful, do feature selection.

### Feature Engineering

Make a problem easier by expressing it in a simpler way. Make the latent manifold smoother, simpler, better-organized.

### Using early stopping

EarlyStopping callback, stop the model as soon as validation metrics have stopped improving.

### Regularizing your model

Set of best pratices that actively impede the model's ability to fit perfectly to the training data. Makes the model simpler and regular.

- reducing the network's size
- adding weight regularization: principle of **Occam's razor**
  - *"Given two explanations for something, the explanation most likely to be correct is the simplest one - the one that makes fewer assumptions."*

Flavors:
- L1 regularization: cost added is proportional to the absolute value of the weight coefficients.
- L2 regularization: cost added is proportional to the square of the value of the weight coefficients. Is also called of *weight decay*.

- dropout: is one of the most effective and most commonly used regularization techniques for NN (Geoff Hinton). Randomly dropping out (setting to zero).
  - output = [1, 2, 3, 5, 6]
  - output_dropped = [0, 2, 3, 0, 6]
  - The dropout rate is the fraction of the features that are zeroed out, **usually set between 0.2 and 0.5.**


# Chap 6

Universal step-by-step blueprint that you can use to approach and solve any machine-learning problem.

Broadly structured in three parts:

- Define the task: understand the problem domain and the business logic underlying.
- Develop a model
- Deploy the model

## Define the task

- What will your input data be? What are you trying to predict?
- What type of ML task are you facing?
- What do existing solutions look like?
  - precision of the currently method utilized.
  - like a research on academia.
- Are there particular constraints you will need to deal with?
- Hypothesis we are mading:
  - our targets can be predicted given your inputs
  - data that's available is sufficiently informative to learn.

### Collect a dataset

Collect and annotate on supervised learning, this is very time consume.

There is some companies that annotate data to use in ML.

If these data can be annotate locally, do we have people specilized to so? We can construct a platform to data-annotation, like a Shiny app.

### Beware of non-representative data

Related to the way you collect the data. Needs to have the data that you will encounter on real world.

Important concept:
- **Concept Drift**: The production data change over time, causing model accuracy to gradually decay. We need to be aware of this because is where we'll need to retrain the model on new data.

So, is very important to keep collecting data about the model you're using in production. 

### Understand your data

Do not treat your data as blackboxes:
- If your data includes images or natural-language, take a look at a few samples directly.
- If contains numerical features, it's a good idea to plot the histogram of feature values...
- If locations, plot it on a map
- Are samples missing values?
- classificatio? Print the number of instances of each class in your data
- Check for target leaking

### Choose a measure of success

Do we have a lot of metrics, you can look on kaggle competitions to get insight of some metric.


## Develop a model

Only one step in the machine learning workflow! The easiest part!


### Prepare the data

Each type of data needs a specific way to deal with. But some pre-processing steps are very common between all types of data.

- vectorization
- value normalization
- handling missing values: consider replace missing value with the average or median value for the feature in the dataset.
  - you can also train a model to predict the missing value based on some other variables.
  - 