# Problems

## Single Feature

In the new python package feature_experiments, create a new module feature_problem.py, which should have a class SingleFeatures which inherits from problem. You should accept device as part of the constructor args like the other Problems. It should take a constructor argument d which is the dimensionality of the problem, it should also take a number of features f. Each feature f is encoded as a specific direction in the input space d. These features will not exist on any particular basis. 

We require f <= d. Let the f features be the f rows of the matrix Q ∈ R^{f×d}, where Q is a random matrix with orthonormal rows (when `is_orthogonal=True`, the default). Alternatively, when `is_orthogonal=False`, the rows of Q are random unit-norm vectors generated with rejection sampling to ensure they are not too close (dot product between any two features must be <= cos(2π/d)). The num_class (number of classes) is exactly f = num_class.

We need to implement `generate_dataset`, every example has exactly 1 and only one feature (and therefore a single class). We should be as class balanced as possible. We could achieve this by stacking `I_f` identity matrices on top of each other, specifically `ceiling(n / f)` identity matrices, and then truncating the number of rows to n, so we have xstandard ∈ R^{n×f}. Finally the matrix multiplication x = xstandard @ Q gives us the (n, d) inputs. The labels y should exactly correspond to the feature. It should be clear what to do with the shuffle flag from the other problems. We do not need to use the clean_mode flag as there is no noise yet. In this case, center_indices is the same as y, we can make a copy of it and return it.

### Percent Correct

We want to introduce some noise to the problem, optionally. Much like the constructor arg n_per_f we have an optional list percent_correct_per_f, which also must be of length f if it is specified. Each element is a number between 1/f <= pc_j <= 1.0. When it is specified, for all examples from a given feature, we "flip" the exact percentage 1-pc_j (where j is the jth feature). If the label is flipped the jth feature gets the j+1th label instead (wrapping around to feature 0 for the last feature). So if pc_j=0.7 and feature j has 20 data points we flip (1 - pc_j) * 20 = 0.3 * 20 = 6 data points to class j+1.

### Noisy d

We will also have an optional constructor argument `noisy_scale = None`. If it is specified, we will create f completely random unit vectors of length d (sampled from standard Gaussian and normalized to unit length, using the generator for reproducibility). Each will then be multiplied by `noisy_scale` (so if it was 1, these random vectors would all stay unit vectors). If it is specified, the full dimensionality for the problem will be 2*d (i.e., the `d` property returns 2*d). For every sample generated, a vector will be selected uniformly at random from the f scaled random vectors (independent of which feature the sample belongs to, providing no class-discriminative signal), and that will be concatenated onto the actual feature vector x (so that it now has 2*d dimensions).

## Kaleidoscope

Let's create the Kaleidoscope problem, we have:
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

# RegAdamW

RegAdamW is a new optimizer which implements a version of AdamW that applies L2 regularization much more effectively to large neural networks. It is implemented in feature_experiments/optimizer.py. We will describe RegAdamW mathematically, then get into the specific implementation details. Recall that AdamW has:

1. A first moment g (this is just the gradient)
2. A second moment g^2
3. A moving first moment m
4. A moving second moment v

with the update given by:
W - lr*(m / (root(v) + adam_eps) + wd)
(where wd is the weight decay as per Adam W) 

We don't want to change the update rule but rather how the raw first and second moments are calculated. The second moment should be replaced with the sum across the per sample gradient squared, `g2_per_n`. That is, suppose g_out [batch_size, d1] is the backpropagated gradient with respect to the output of the Linear module (i.e., ∂L/∂(xW^T)), and x [batch_size, d0] is the input to the linear module. Then we have:
`g2_per_n = g_out.t() ** 2 @ x ** 2`

The first moment should also change, for each weight we want to scale it by the square root of the number of examples that were not zero-ed out by ReLUs, normalized by batch size. More specifically, we have `non_zero_count = ((g_out != 0).t() @ (x != 0))`. The comparison to zero is exact (not threshold-based).

So to clarify, let g be the standard gradient ∂L/∂W (i.e., `g = g_out.t() @ x`):
1. The first moment must be replaced with `(non_zero_count ** 0.5 / batch_size) * g`
2. The second moment must be replaced with `g2_per_n`
3. Otherwise, the implementation is the same as AdamW

## Implementation Details

We assume all Linear modules have bias=False (which is the case throughout the codebase).

We can use gradient hooks to calculate and store the tensors `g2_per_n` and `non_zero_count` on each parameter. Then we can implement a new Optimizer RegAdamW in `feature_experiments/optimizer.py`, to calculate the new moments and apply the update.

We use two hooks on each Linear module:
1. A forward hook to store the input tensor x on the module (e.g., `module._reg_input = x`)
2. A `register_full_backward_hook` to compute `g2_per_n` and `non_zero_count` using `grad_output[0]` as g_out

These two fields (`g2_per_n` and `non_zero_count`) can be stored directly on the nn.Parameter for use by the optimizer.

## Use

Notice config.py, this has the flag is_adam_w, this should be replaced with the optimizer string field with "adam_w" as the default, the other options should be "sgd" and "reg_adam_w". The optimizer field should be validated in Config.__init__. Firstly update all existing code which uses it to switch to the string parameter instead.

**Important constraint:** When `optimizer="reg_adam_w"`, the config must have `learnable_norm_parameters=False`. If `learnable_norm_parameters=True` with `optimizer="reg_adam_w"`, raise a ValueError in Config.__init__.

For the "reg_adam_w" path, it is unsupported for the empirical_runner pathway (raise an exception). However if specified for single_runner then it should be used. It is of course important on this pathway to register the hooks on the model. We can support all models, because we need not touch the model code directly to register the hooks.

