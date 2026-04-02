### Proposing an experiment

Inspired by \[Deng et. al, 2020\] and using the results in \[Soudry et. al 2017\], we make a more general, but less precise application of the “support vector theory” of double descent.

For problems where there is some entropy in the target (all real problems), some unexplainable variation, the next token / correct class is not a certainty. There must be some points x in the training set which do not correspond to the most likely token/class as prescribed by the true distribution p(x). If a model is to perfectly separate the data, these points can only be separated by spurious features. For model classes which have some success generalizing, we propose:

1. For a model that is well parameterized in the classical regime, around the classical minimum of the bias-variance tradeoff, the model is **composed** more of the likely points, less of the unlikely points, in proportion to their likelihood.

2. For model capacities at the interpolation threshold, the gradients are dominated by those unexplainable points, using spurious features. This means the model is **composed** primarily of these unexplainable points. The model’s finite capacity is heavily used to fit these points; they are the model’s support vectors. The fact that these are the precise points that are the worst at reflecting the true distribution, distorts the model and degrades generalization significantly.

3. As model capacity increases, the separation of every point becomes equally trivial, every point becomes a support vector. The model is **composed** equally of all training points. The model fits to all features, but there is a normalization effect across spurious features, and consequently variance decreases and generalization improves.

Of course the loaded word here is “composed”. A training point’s contribution is not simply the sum of its gradients.  [\[Yeh & Kim et. al, 2018\]](https://arxiv.org/abs/1811.09720) provide an accurate way to decompose a model into a linear combination of kernel functions based on the training points. Using their technique (which we describe in detail in Appendix X), and examining the gradients, we can test the three conclusions above.

We’ll start with a simple problem of our own construction. Knowing the distribution of the data, we can test the finer points of the theory. Then we’ll test our Resnet18 and transformer model, and see if the results hold.

#### Training point Decomposition

[\[Yeh & Kim et. al, 2018\]](https://arxiv.org/abs/1811.09720) prove a model can be decomposed into its training points when:

1. The model’s final layer is a sum of the output of earlier layers (basically any neural net)  
2. The model is trained, or at least has its final layer fine-tuned with L2 regularization.  
3. That model is trained / fine tuned model is at a stationary point with respect to the loss, L(x, y) \+ W2

The criteria are very practical, most fully trained deep learning models would meet them except for the fact that the optimizers we use, whether SGD or AdamW do not reach exact stationary points. Irrespective of whether L2 was used in the main training run, we can fine tune the same loss \+ L2 just on the final weights, using a more precise optimizer like line search to reach a stationary point. Once we have that, we may express the model’s logit for the jth class / token, on an unseen training point xt as:  
fj(xt) \= \-12ni=1ndL(xi, yi)dfj(xi)(xi)T(xt)  
where (x) is the value of the network up until the layer before the logits, we point the reader to the [\[Yeh & Kim et. al, 2018\]](https://arxiv.org/abs/1811.09720) for the full derivation.

Figure X: The loss for the original and fine-tuned resnet18 and transformer. The fine-tuning step does change our model and will affect the loss. Given L2’s rotational invariance, and the fact we only tune the final layer, we expect only the scale of the loss to change. We show the log loss once again for the fine-tuned model, against the original, demonstrating nothing has fundamentally changed with respect to the double descent.