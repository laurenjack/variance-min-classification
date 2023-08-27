# variance-min-classification

Attempting to directly minimize the bias variance trade off.

# Introduction

Most people who work in Machine Learning are aware of the of the [bias-variance trade off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). Even if they don't recall the term, show them the picture below and they're immediately familiar with the concept:

![Bias-Variance Tradeoff](img/3_graph.png)

I'm going to lay out the obvious, well-known aspects of the bias-variance trade-off, then dive deeper on model variance. IMO, we don't think about this variance deeply enough, or at least that has been my experience in industry. Model variance is misunderstood. The training set for any target prediction problem, is just n samples [(x1, y1), (x2, y2) ... (xn, yn)] drawn from the random variables (x, y). The key point about variance is that a model trained on the first n samples will be different a model trained on the next n samples. However, we never observe this difference, because naturally we train on all the data avaiable. In other words, the model parameters and model predictions, are random variables:

- ![x_y] &nbsp; The input and target, random variables

- ![theta] &nbsp; The model parameters, a random variable

- ![y_hat] &nbsp; The prediction, also a random variable

- ![var] &nbsp; Model variance

The strange thing about machine learning as a field, is that we do not measure, nor attempt to minimize the variance in our models. Bias and variance are both present in any loss function, here is the Mean squared error, decomposed into bias and variance:

- ![mse]

- ![mse_bv]

- ![mse_empirical] - The empirical loss function.

We train a model by minimizing the empirical loss function. The empirical loss function is mischevious, because at first glance it looks like it should have a role in minimizing variance. However minimizing this reduces bias, but continuously increases variance (given the same initial parameterization). The more expressive the model, the more bias can be reduced, but the more variance is increased, as should be obvious to the reader. Variance is not reduced by minimizing the empirical loss function, it is reduced primarly with 3 techniques:

1. **More Data** The most straight forward variance reduction technique. More data forces variance down, as there are more examples per parameter, meaning less overfitting can take place. Often, there isn't much a practitioner can do to change the volume of data they have. However, over the long term, the reduced cost of computing, and (more recently) the increased returns / funding to ML has meant training on much larger datasets has become possible.

2. **Validation, Test sets, online testing + Graduate Student Descent** Data does not continuously reduce variance on it's own, the degree to which additional data reduces variance, is dependent on the choice of model, futhermore some models are just better (less bias and less variance). Discovering modelling paradigms which allow for increased complexity (reduced bias) while keeping variance low is non-trivial (to say the least). The ML field's 70 year history has been a continous process of trial and error with empirical feedback, targeted at solving this problem. Let's call this process **Graduate Student Descent (GSD)**, not to belittle the progress, but rather to highlight that it's optimization process for reducing variance.

GSD has lead to the discovery of modelling paradigms which achieve lower bias and variance (e.g. [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting#:~:text=Gradient%20boosting%20is%20a%20machine,which%20are%20typically%20decision%20trees.)), and paradigms such as deep learning which lower bias more than ever before, while maintaining an acceptable level of variance. Large Language Models, pair better with "more data" than any previous paradigm. Scaling them up + adding more data consistenly leads to lower variance (OpenAI document this phenomenon [here](https://arxiv.org/abs/2005.14165)).

3.**Regularization methods** This is the only technique of the three that attempts to reduce variance in the training process. L2 regularization is the most widely used regularization method. Simple and effective, it modifies the empirical loss function, setting a small penalty on parameter size. This biases the weights towards zero, such that only parameters with sufficient explanatory power remain non-zero. Other methods include early stopping, and dropout (neural network specific)

The key observation I'm trying to highlight here, is that the training process is essentially a bias reduction process, and that the level of variance is more or less implicit in the choice of model architecture. Yes regularization (3) is a tool which allows us to trade higher bias for lower variance, but relevative to the amount of data (1) and choice of model (2) the impact on variance is small.

What about hyper-parameter optimization, bagging or cross validation, or any other variance reduction techniques I've missed? Thses are all part of GSD, where we build a model and check it's performance on the validation set, then try again. The point is, there is very little variance minimization inside the main optimzation, the training process.

If I take a large neural net, and train it on a small dataset, it will ruthelessly over fit. It will fit the training examples perfectly but the validation examples poorly, i.e. it will have high variance. To minimize variance, I have to try different, smaller architectures, like they did in this [paper](https://arxiv.org/pdf/2003.12843.pdf). Futhermore, it is not clear that an optimum trade-off between bias and variance is reached.

Stated more directly, **where is the modelling paradigm, which trades off between bias and variance in the training process?**

# Measuring Variance

Suppose we could take m \* n samples {[(x11, y11), (x21, y21) ... (xn1, yn1)], ... [(x1m, y1m), (x2m, y2m) ... (xnm, ynm)]} where we had m training sets, each of sample was of size n. Suppose we then fit a model to predict y from x from each of them. Then with a sufficently large m, and a validation set [(x1, y1), (x2, y2) ... (xn, yn)], we would have a good empirical estimate for model variance. Let:

- ![y_hat_j] &nbsp; The prediction of the jth model on the ith element of the validation set

- ![y_hat_bar] &nbsp; The mean prediction on the ith example of the validation set (mean over m models)

- ![var_empirical] &nbsp; Empirical model variance (over the validation set)

By breaking larger datasets into m subsets, we can experiment with modelling approaches that attempt to minimize variance and directly measure their ability to do so. We'll propose some approaches for minimizing variance in the training process. Then, we will test these out using the CIFAR10 dataset, as well as some synthetic datasets that are designed to induce model variance.

# Minimzing Variance

## Variance - the Dissipearing Signal.

Our first instinct is to minimize the empirical vairance equation, by adding it to the empirical loss function, to produce a new total loss:

- ![total_empirical]

Each model j is trained on it's own mutually exclusive subset of the data, as data is the driver of variance. However, each model also fits to the mean prediction of the m models, over a separate "validation set" which we will call the variance set. I use quotes on "validation set" because even though we don't use the targets in training, we're still training on the inputs, so it's no longer a validation set in the strict sense. So to be clear:

- Each model j is optimized on it's own training set, AND

- Each model j is optimized on the same variance set, because each model's prediction is in the mean prediction term: ![y_hat_bar_single]

In some sense this works, the second term (variance loss) does cause the model to generalize better. However, perhaps you can already see the issue with the second term, to put it simply we're just making each model more like each other. Just like the empirical loss functon, minimizing this term doesn't really minimize variance on unseen data, the models just overfit to each other. We see this when training on three different objective functions:

Minimizing equation 1 does lower the variance, and it does so using unlabelled examples. This is essentially semi-supervised learning, where models are trained on the predictions of unlabelled training examples, as they do in this [paper](https://arxiv.org/abs/1905.00546). Perhaps variance reduction is part of the explination for why semi-supervised learning works. At any rate, in this instance we'd be better of just using 2x the data on the empirical loss functon, as shown in graph 3 above.

The core of the problem, is that there are no free lunches when it comes to the bias variance trade-off. In equation 1, we are not constraining the model in any way. For a given architecture, to obtain lower variance we need to accept a greater level of bias. L2 regularlization makes this choice by biases the weights towards zero, reducing model variance.

## L2 Regularization

The gradient for a single weight in a neural network that is trained L2 regularization looks like this:

- ![g] Where L is the loss function w is the weight and r is the L2 weight decay constant. The first term is the gradient from the empirical loss function, the second is the regularization gradient.

Looking at the equation we see the trade off in action, if the gradient from the empirical loss function is not large enough, then the regularization gradient will bias the weight close to zero. In some sense, only weight increases that induce a significant reduction in the loss are allowed (this isn't the only reason regularization works[<sup>1</sup>](#reg_reasons)).

Why is this a useful bias to have? Because when all features are normalized (e.g. scaled between -1.0 and 1.0), genuine signals tend to be larger than noise. To intuite why, let's consider a simple binary classification problem, with 2 inputs feature x1 and feature x2:

- Feature x1 accurately predicts the class 80% of the time. 80% of the time; when it is positive the class is class A, and when it is negative it is class B.
- Feature x2 is just noise.
- Fit with a softmax with 2 outputs, one for class A and class B, fully connected to the input layer with no biases. Vector wA for class A, and vector wB for class B

DRAW PICTURES

The expected value of the learned parameters, or in other words the optimal solution (over unseen data):

- ![wa]
- ![wb]

Where c is a positive constant, it's exact value is determined by the network upstream of x (in this case , just the softmax and loss function). What should be intuitive is that output A has a positive weight c on feature 1, output B has a negative weight c on feature 1, and the weight of both on feature 2 should ideally be zero.

Let ![sigma] be the variance of feature i, then we can express the variance associted with the training process:

- ![var_1]
- ![var_2]

Where k, like c determined by the network upstream of x. This variance means the learned parameter always has some degree of error:

- ![wa]

The optimal solution leads to a classification accuracy of 80%. Deviations in wA and wB from the optimal from will lead to a lower accuracy, as the size of the error term increases.

Let us draw some conclusions from the value of wA (the same apply to wB).

1. More data reduces variance. We can see that as n increases, variance

2. Scaling the inputs matters. If feature 2 was on average 10x the magnitude of feature one then we would need more data to achieve the same classifcation accuracy[<sup>2</sup>](#scaling_gradient). The scaling ensures: ![sigma_approx]

3. As the size of the error term approaches the magnitude of c, overfitting becomes chronic and the model fails to generalize.

What is missing above is a conclusion about k. In this example problem, one can infer that ![ck], as shown empirically below. The size of k, relative to c, determines whether we can generalize or not for a given n. Calling this term k hides the complexity of determing it on real problem (where k is not knowable). Nonetheless, the equation above serves as a useful model for thinking about variance as we progress.

It was already obvious from the pictures, but given that ![sigma_approx] and for this particular problem ![ck], for all but the smallest n, without any regularization this model will reach a classification accuracy of 80%.

k and n determine whether we can generalize successfully. - Given that our model can approximate the desired function and gradient descent can find that approximation. In this problem: ![ck]

To generalize as best as possible w.r.t classification accuracy (80%), an optimal model would have the weight vectors [wA, 0] and [-wB, 0]. Where wa > 0 and wb > 0. Let us consider the relative magnitudes of the gradient on feature 1, and the gradient on feature 2:

- ![e_feature1_grad_1] The gradient on

Feature 1:

Suppose feature 1 has an average magnitude s

We'll look at a simple example to build intuition, then return to the general point. Consider a binary classification problem with 2 input features. Let's suppose the first feature is a reasonable predictor of the class, suppose that 80% of the time, positive values indicate class 0, negative values indicate class 1. The second feature is completely random w.r.t class. Feature one contains a genuine signal (with some noise), feature 2 is pure noise. We'll fit the problem with a single softmax layer ie:

- y_hat = softmax(xW)

Consider

<a name="reg_reasons"></a>[^1]: This isn't the only reason L2 regularization works, another reason (which might be isomorphic) is that the nodes are encourage to learn a single feature, rather than 10 nodes learning a weaker version of the same feature.

<a name="scaling_gradient"></a>[^2]: Features of different scale also make gradient descent and related techniques more difficult, because the learning rate would have to accomodate different scales of gradients.

[x_y]: https://chart.apis.google.com/chart?cht=tx&chl=(x_i%2Cy_i)
[theta]: https://chart.apis.google.com/chart?cht=tx&chl=\theta
[y_hat]: https://chart.apis.google.com/chart?cht=tx&chl=\hat{y}=f(x%2C\theta)
[y_hat_j]: https://chart.apis.google.com/chart?cht=tx&chl=\hat{y}_{ij}=f(x_i%2C\theta_j)
[var]: https://chart.apis.google.com/chart?cht=tx&chl=VAR(\hat{y})=E(\hat{y}-E(\hat{y}))^2
[mse]: https://chart.apis.google.com/chart?cht=tx&chl=L=E(\hat{y}-y)^2
[mse_bv]: https://chart.apis.google.com/chart?cht=tx&chl=L=E(\hat{y}-\bar{y})^2%2BE(\hat{y}-E(\hat{y}))^2
[mse_empirical]: https://chart.apis.google.com/chart?cht=tx&chl=L=\sum_{i=1}^{n}(\hat{y}-y_i)^2
[var_empirical]: https://chart.apis.google.com/chart?cht=tx&chl=var(\hat{y})=\sum_{i=1}^{n}(\hat{y}_{ij}-\bar{\hat{y}}_i)
[total_empirical]: https://chart.apis.google.com/chart?cht=tx&chl=L=\sum_{j=1}^m\sum_{i=1}^{n}(\hat{y}_{ij}-y_{ij})^2%2B\sum_{i=1}^{n}(\hat{y}_{ij}-\bar{\hat{y}}_i)
[fx]: https://chart.apis.google.com/chart?cht=tx&chl=f_j(x_i)
[y_hat_bar]: https://chart.apis.google.com/chart?cht=tx&chl=\bar{\hat{y}}_i=\frac{1}{m}\sum_{j=1}^{m}\hat{y}_{ij}
[y_hat_bar_single]: https://chart.apis.google.com/chart?cht=tx&chl=\bar{\hat{y}}
[g]: https://chart.apis.google.com/chart?cht=tx&chl=g=\frac{dL}{dw}%2Brw
[sigma]: https://chart.apis.google.com/chart?cht=tx&chl=\sigma

[wa]: https://chart.apis.google.com/chart?cht=tx&chl=E(w_A)=[c,0]
[wb]: https://chart.apis.google.com/chart?cht=tx&chl=E(w_B)=[-c,0]
[var_1]: https://chart.apis.google.com/chart?cht=tx&chl=var(w_{A1})=var(w_{B1}=k^2\frac{\sigma_1}{n}
[var_2]: https://chart.apis.google.com/chart?cht=tx&chl=var(w_{A2})=var(w_{B2}=k^2\frac{\sigma_2}{n}
[wa_error]: https://chart.apis.google.com/chart?cht=tx&chl=w_A=[c\pok_1\sqrt{\frac{\sigma_1}{n}},k_2\sqrt{\frac{\sigma_2}{n}}]
[sigma_approx]: https://chart.apis.google.com/chart?cht=tx&chl=\sigma_1\approx\sigma_2
[ck]: https://chart.apis.google.com/chart?cht=tx&chl=c\approxk

[feature1_gradA]: https://chart.apis.google.com/chart?cht=tx&chl=\frac{dL}{dw_A}=\frac{dL}{da}*\frac{da}{dw_A}
[feature1_gradB]: https://chart.apis.google.com/chart?cht=tx&chl=\frac{dL}{dw_B}=\frac{dL}{da}*\frac{da}{dw_B}
[e_feature1_grad_2]: https://chart.apis.google.com/chart?cht=tx&chl=(\frac{dL}{da})*E(\frac{da}{dw_A})
