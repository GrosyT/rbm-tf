# %%DBNSETUP creates a propr dbn struct
# %     INPUT
# %        sizes : A vector with hidden layer sizes
# %            x : used to specify size of first hidden layer
# %         opts : a struct with options see dbncreateopts
import numpy as np

def dbnsetup(sizes, x_train, opts):
    n = x_train.shape[1]
    dbn_sizes = sizes

    class Dbn:
        assert isinstance(dbn_sizes, object)
        sizes = [n, dbn_sizes]
        n_rbm = len(sizes)

        #% create weight initialization function
        if opts.init_type == []: #'empty' #isinstance(opts.init_type, empty)
            initfunct = opts.init_type
        elif opts.init_type.isalpha():
            if opts.init_type == 'gauss':
                initfunct = lambda m,n: np.random.normal(0,0.1,m,n)


        @property
        def myfunc(self):
            return self

    dbn = Dbn
    return dbn, dbn_sizes                                  #TODO: pass sizes to dbn class better

