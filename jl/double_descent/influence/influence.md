The Support Vector Theory of Double Descent
For most of the real world distributions we care about there is variation we cannot explain. Next-token prediction is a great example of this. Knowledge of the previous tokens can only tell you so much about the next, there is some irreducible entropy H(xt|x0:xt-1) in the next token.

For a simple dataset like CIFAR10, the fact that a Resnet can generalize with near a 100% accuracy demonstrates the irreducible entropy in CIFAR10 labels is near zero. The 15% mislabelling 15% is useful because it allows us to control this irreducible entropy. We know that this 15% cannot be explained, fitting them is certainly overfitting.

For models around the interpolation threshold, we hypothesize that these unexplainable points have larger gradients for more epochs, and in particular larger gradients towards the end of training. Therefore, the points which are the most surprising (impossible to explain even) have a disproportionate influence on the model.


However as model capacity increases, it becomes increasingly trivial to overfit every point. The gradients for each point equalizes, as the exponential term (yi -qi) vanishes more equally across training points. The influence of every training point equalizes, causing the error and loss to fall. If the classical minimum was ill-specified, this overparameterization can lead to a better minimum.
Influence
So far it all sounds good, but we have been imprecise about this concept of influence. It is easy enough to think about the gradient per point, but a model is not the sum of its per-point gradients. In fact, the magnitude of the per-point gradient is not even indicative of its influence on the actual gradient. In the classical regime, error term (yi -qi) remains large for the entire training run, the whole premise is those errors cancel out.

Inspired by [Deng et. al, 2020] and using the results in [Soudry et. al 2017], we make a more general, but less precise application of the “support vector theory” of double descent.

For problems where there is some entropy in the target (all real problems), some unexplainable variation, the next token / correct class is not a certainty. There must be some points x in the training set which do not correspond to the most likely token/class as prescribed by the true distribution p(x). If a model is to perfectly separate the data, these points can only be separated by spurious features. For model classes which have some success generalizing, we propose:

For a model that is well parameterized in the classical regime, around the classical minimum of the bias-variance tradeoff, the model is composed more of the likely points, less of the unlikely points, in proportion to their likelihood.


For model capacities at the interpolation threshold, the gradients are dominated by those unexplainable points, using spurious features. This means the model is composed primarily of these unexplainable points. The model’s finite capacity is heavily used to fit these points; they are the model’s support vectors. The fact that these are the precise points that are the worst at reflecting the true distribution, distorts the model and degrades generalization significantly.


As model capacity increases, the separation of every point becomes equally trivial, every point becomes a support vector. The model is composed equally of all training points. The model fits to all features, but there is a normalization effect across spurious features, and consequently variance decreases and generalization improves.

Of course the loaded word here is “composed”. A training point’s contribution is not simply the sum of its gradients.  [Yeh & Kim et. al, 2018] provide an accurate way to decompose a model into a linear combination of kernel functions based on the training points. Using their technique (which we describe in detail in Appendix X), and examining the gradients, we can test the three conclusions above.

We’ll start with a simple problem of our own construction. Knowing the distribution of the data, we can test the finer points of the theory. Then we’ll test our Resnet18 and transformer model, and see if the results hold.
Training point Decomposition
[Yeh & Kim et. al, 2018] prove a model can be decomposed into its training points when:

The model’s final layer is a sum of the output of earlier layers (basically any neural net)
The model is trained, or at least has its final layer fine-tuned with L2 regularization.
That model is trained / fine tuned model is at a stationary point with respect to the loss, L(x, y) + W2

The criteria are very practical, most fully trained deep learning models would meet them except for the fact that the optimizers we use, whether SGD or AdamW do not reach exact stationary points. Irrespective of whether L2 was used in the main training run, we can fine tune the same loss + L2 just on the final weights, using a more precise optimizer like line search to reach a stationary point. Once we have that, we may express the model's j-th class / token logit Φ_j on a (possibly unseen) point x_t as:

  Φ_j(x_t) = -1/(2λn) · Σ_i (∂L(x_i, y_i)/∂Φ_j(x_i)) · f(x_i)^T f(x_t)

where, matching [Yeh & Kim et. al, 2018]:
- Φ(x) is the logits vector — the network output up to and including the final linear layer.
- f(x) is the feature vector — everything in the network up until the final linear layer.

We point the reader to [Yeh & Kim et. al, 2018] for the full derivation.
