# Problems

## Single Feature

In the new python package feature_experiments, create a new module feature_problem.py, which should have a class SingleFeatures which inherits from problem. You should accept device as part of the constructor args like the other Problems. It should take a constructor argument d which is the dimensionality of the problem, it should also take a number of features f. Each feature f is encoded as a specific direction in the input space d. These features will not exist on any particular basis. 


Let us suppose that f <= d. Let the f features be the f rows of the matrix Q  Rfd, where Q is a random matrix with orthonormal rows. The num_class (number of classes) is exactly f = num_class. Now let's suppose f > d, the feature vectors (rows of Q) should form points on a Unit Norm Tight Frame (UNTF).

We need to implement `generate_dataset`, every example has exactly 1 and only one feature (and therefore a single class). We should be as class balanced as possible. We could achieve this by stacking `I_f` identity matrices on top of each other, specifically `ceiling(n / f)` identity matrices, and then truncating the number of rows to n, so we have xstandard Rnf . Finally the matrix multiplication x = xstandardQ gives us the (n, d) inputs. The labels y should exactly correspond to the feature. It should be clear what to do with the shuffle flag from the other problems. We do not need to use the clean_mode flag as there is no noise yet. In this case, center_indices is the same as y, we can make a copy of it and return it.

## Kaleidoscope

Let’s create the Kaleidoscope problem, we have:
d - the number of inputs
[C_0, C_1, C_2, ...CL-1] - the centers, where C_{L-1} is the number of classes.

We require that \foreach 0 <= l < L, C_l <= d, so that we can generate orthognal features at each layer. Similar to the SingleFeature problem, each layer generates C_l random unit vectors in d dimensional space that we orthonormalize (there is no UNTF path). Let us represent this as the matrix Q_l \elementOf R^{C_{l}×D}, Q would now be a list of such matrices. We should take a flag is_standard_basis=False, when true those random directions should always be drawn from the standard basis.

Now for the sampling. We shall iterate over the l centers, summing those. Specifically, let an individual vector x be:

x_i= Q_0[c_{0i}] + \sum_{l=1}^{L-1}(2^{-l}Q_l[c_{li}])  Where c_{li} is the randomly chosen center for point i at layer l chosen uniformly over the integers: 0<= c_{li} < C_l, that is the selected direction. The c_{(L-1)i} corresponds to the class label y_i. Ideally we can generate the entire matrix of data points at once, as opposed to constructing each vector individually.

### Class Balance

For a given problem layer, the samples should be balanced across centers as possible. However, it is important that from layer to layer there is no correlation of centers.

### Center Indices 

The Kaleidoscope problem needs to return a list of center indices, one for each layer of the problem. Each element should be of shape [n,] and the elements of that tensor should correspond to center index at that layer for the given data point. 

### Additional details

- Let's make sure we refactor things such that the problems share common logic

## TiltedKaleidoscope

Let us create a new class TiltedKaleidoscope, reuse what is applicable from Kaleidoscope. It will be similar in that we have a list of centers, except now the last center does not correspond to the class, it is instead the case that TiltedKaleidoscope is always a binary classification problem, i.e. 2 classes. Like Kaleidoscope, centers should be constructed in the same fashion (don't make a center for the 2 classes though). Each number of centers must be divisable by 2. In the constructor, for each layer, exactly 50% of the classes should be marked as tilted to class 0, and the other 50% should be marked as tilted to class 1.

The tilt_increment for the instance should be caluculated as 0.5 / len(centers). The generation works as follows, we should have exactly 50/50 class balance. We start from the class, consider a single point that is assigned class 0. We have a `tilt` per layer, that is just `tilt = 0.5 + tilt_increment * (layer_index + 1)`, such that the last layer is always 1. (In fact, in the last layer we should deterministically assigned classes)

At each layer (prior to the last), the per center probability, for centers tilted to 0 is tilt / (num_center_this_layer / 2), the probability it is assigned to centers tilted to 1 is (1 - tilt) /  (num_center_this_layer / 2). We need not center balance here just generate which center is chosen based on those probabilities. The same applies to class 1, for centers tilted to classs 1.

## Feature Combinations 

Let us define a problem FeatureCombinations, it takes an integer argument num_layers as an argument. Let d = 2 * 2^num_layers. Let atomic_d = 4 (the dimensionality of each atomic subsection).

### Atomic Layer (Layer 0)

The input space is split into num_subs = d // atomic_d = 2^num_layers // 2 atomic subsections. Each atomic subsection spans atomic_d = 4 dimensions of the input space.

Each atomic subsection has its own orthonormal basis of 4 random feature vectors (a 4×4 orthonormal matrix Q_i for subsection i). When we choose an atomic feature for subsection i, we are choosing 1 of those 4 basis vectors uniformly at random. So then at layer 0 we have num_subs atomic features for each example, one per subsection.

### Subsequent Layers (Consolidation via Random Assignment)

Each subsequent layer has half the number of subsections as the previous layer. Each subsection at layer L is composed of 2 contiguous subsections from layer L-1.

Since each prior subsection has 4 possible feature values, two prior subsections have 4×4 = 16 possible combinations. We represent a combination as (i, j) where i, j ∈ {0, 1, 2, 3}.

The consolidation uses **completely random assignment** to partition the 16 combinations into 4 groups of 4:

1. Generate a random permutation of the 16 combinations
2. Assign the first 4 to consolidated feature 0
3. Assign the next 4 to consolidated feature 1
4. Assign the next 4 to consolidated feature 2
5. Assign the last 4 to consolidated feature 3

This does NOT guarantee linear separability - the assignment is completely random. A different random permutation is generated for each subsection at each layer.

### Final Layer

This continues until the last layer, where there is just a single subsection with 4 possible consolidated features. At the final layer, 2 of the 4 features are randomly assigned to class 0, and the other 2 are assigned to class 1. This allocation determines the class for all generated points.

### Constructor Arguments

- `num_layers`: Number of layers in the hierarchy (must be >= 2)
- `random_basis`: If True, rotate the final x with a random orthonormal basis (see other problems)
- `device`: torch device
- `generator`: optional random generator for reproducibility

### Data Generation

Generate n datapoints by:
1. Randomly generating the num_subs atomic features (each uniformly from {0, 1, 2, 3})
2. For each atomic subsection, look up the corresponding basis vector from Q_i to construct x
3. Deduce the consolidated features layer by layer using the pre-computed consolidation mappings
4. At the final layer, map the final feature to the class using the pre-computed class assignment
5. If random_basis=True, apply the random rotation to x

Return x, y, and center_indices_list (a list of tensors tracking features at each layer).



# Logit Prior

Let us define a new boolean parameter on the @config is_logit_prior=False. If it is False the behavior is as it is now. If it is true, we are going to do use a different set of atomtic modules, and compute a different kind of backprop.

First of all we will need to create a new python module logit_prior, it should have a class LinearLP. This is a class that extend pytorch's linear, we will be implementing a customer autograd function in this module too, and this class LinearLP should call that in it's forward method, such that it implements that custom autograd function. LinearLP should not support biases, just a weight matrix. 

Now the autograd function, the forward will be just the same, on the backwards pass we are going to leave the backpropagated gradient alone, but we are going to change the way that we compute the weight gradient. Let dF_dz be the tensor og the loss w.r.t the output of the linear unit z. dF_dz is of shape [m, d_out] (where m is the batch size here). we shall separate it into two tensors slicing along the batch dimension, dL_dz = dF_dz[:m // 2] and dZ_dz = dF_dz[m // 2:], we can assume m is always even.
We will then compute the tensor for_weight = dL_dz / (torch.abs(dZ_dz) + 1e-8), and use this tensor to compute the weight gradient, dF_dz will be backpropagated to the previous layer as normal.

Now we need to modify @single_runner.py, we have train_once, we need to paths here now _train_once_standard and _train_once_logit_prior. _train_once_standard has the existing, standard training behavior. _train_once_logit_prior should do much the same, and we should re-use common functions, the only difference should be the training step. When we get to the the training step, we should "double the input batch" concantenating the input batch with itself, producing a double length batch dimension. We feed that forward, and then we slice the batch dimension, such that we have two logit tensors of batch size (which are identical by virtue of being copies of the same batch) To the first we apply the standard loss L (whether multi class cross entropy or BCE), to the second we apply the loss Z which is the sum of squared logits, across the logit dimension and mean across the batch dimension. We can then sum the two losses, and now we are ready for backpropagation.

Finally we need to be able to use this LinearLP in models, logit_prior can only be set to True if width_varyer is None, otherwise we should raise, this means that MLP, MultiLinear and Resnet need to take a constructor argument linear, which has a default value of nn.Linear, this argument linear should specify the linear constructor. When logit_prior is True, we need to pass in LinearLP as the linear constructor

All evaluation, whether on the training set or validation set should be carried out normally, on a sinlge batch, rather than a double batch. We only use the "double batch" approach to get the special gradient normalization during training

## RMSNormLP

Similar to LinearLP, we also need a special version of RMSNorm for logit prior training. The logit_prior module should contain a class RMSNormLP that uses a custom autograd function (RMSNormLPFunction). 

RMSNormLP only supports non-learnable weights. In config validation, we must raise an error if is_logit_prior=True and learnable_norm_parameters=True.

The models (MLP, MultiLinear, Resnet, ResidualBlock) take an additional constructor argument rms_norm_class with default value RMSNorm. When is_logit_prior=True, the factory functions pass in RMSNormLP as the rms_norm_class constructor.

The custom autograd function RMSNormLPFunction implements:
- Forward: Standard RMSNorm computation: y = x / rms where rms = sqrt(mean(x^2, dim=-1) + eps)
- Backward: Modified gradient that multiplies through by the "unstable denominator" (rms). Instead of the standard gradient `grad_x = grad_output / rms - x * (grad_output * x).sum() / (d * rms^3)`, we use `grad_x = grad_output - x * (grad_output * x).sum() / (d * rms^2)`. This is the standard gradient multiplied by rms.