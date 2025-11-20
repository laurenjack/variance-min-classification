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

### Additional details

- For now just return None for the center indices.
- We don't class balance here, uniform random is OK
- Let's make sure we refactor things such that the problems share common logic

