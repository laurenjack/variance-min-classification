# variance-min-classification
Attempting to directly minimize the bias variance trade off.

# Introduction

Most people who work in Machine Learning are aware of the of the [bias-variance trade off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). Even if they don't recall the term, show them the picture below and they're immediately familiar with the concept:

![Bias-Variance Tradeoff](img/3_graph.png)

I don't think we think about this concept deeply enough, at least that has been my experience in industry. Specifically, model variance is misunderstood. The training set for any target prediction problem, is just n samples [(x1, y1), (x2, y2) ... (xn, yn)] drawn from the random variables (x, y). The key point about variance is that a model trained on the first n samples will be different a model trained on the next n samples. However, we never observe this difference, because naturally we train on all the data avaiable.

To be more precise, suppose we could take m * n samples {[(x11, y11), (x21, y21) ... (xn1, yn1)], ... [(x1m, y1m), (x2m, y2m) ... (xnm, ynm)]} where we had m training sets, each of sample was of size n. Suppose we then fit a model to predict y from x from each of them. Then with a sufficently large m, we would have a good empirical estimate for model variance. Let:

- ![x_y] &nbsp; The input and target, random variables

- ![theta] &nbsp; The model parameters, a random variable

- ![y_hat] &nbsp; The prediction, also a random variable

- ![var] &nbsp; Model variance

[x_y]: https://chart.apis.google.com/chart?cht=tx&chl=(x_i%2Cy_i)
[theta]: https://chart.apis.google.com/chart?cht=tx&chl=\theta
[y_hat]: https://chart.apis.google.com/chart?cht=tx&chl=\hat{y}=f(x%2C\theta)
[var]: https://chart.apis.google.com/chart?cht=tx&chl=E(\hat{y}-E(\hat{y}))
[fx]: https://chart.apis.google.com/chart?cht=tx&chl=f_j(x_i)
[y_hat_bar]: https://chart.apis.google.com/chart?cht=tx&chl=\bar{\hat{y}}

[^1] I am using capital letters to represent the sample 
