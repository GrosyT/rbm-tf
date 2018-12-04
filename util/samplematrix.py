import numpy as np

def samplematrix(x):
    # SAMPLEMATRIX create a randomized sample from a matrix of probabilities
    # assumes that each row of x is normalized. Samples from each of row
    # of x using uniform distribution
    # [n_samples,n_classes] = size(x);
    # sample = zeros(n_samples,n_classes);
    # r = rand(n_samples,1);
    # for i = 1:n_samples
    #     aux = 0;
    #     for j = 1:n_classes
    #         aux = aux + x(i,j);
    #         if aux >= r(i)
    #             sample(i,j) = 1;
    #             break;
    #         end
    #     end
    # end
    #

    # vectorized implementation
    n_samples, n_classes = x.shape
    sample = np.zeros((n_samples,n_classes))

    r = np.random.uniform(0, 1, (n_samples, 1))
    x_c = np.cumsum(x, 1)
    larger = x_c >= r
    idx = max(larger, [], 1)

    lin_idx = np.ravel_multi_index(np.transpose(n_samples) ,x.shape)
    sample[lin_idx] = 1
    return sample