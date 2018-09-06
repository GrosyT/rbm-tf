# %%DBNSETUP creates a propr dbn struct
# %     INPUT
# %        sizes : A vector with hidden layer sizes
# %            x : used to specify size of first hidden layer
# %         opts : a struct with options see dbncreateopts
import numpy as np



def crbm_init_weights(m, n):                    #todo complete weight init func in cRBM case
    pass

def dbnsetup(sizes, x_train, opts):
    n = x_train.shape[1]
    dbn_sizes = [n, sizes]
    n_rbm_1 = len(dbn_sizes)

    class Dbn:
        assert isinstance(dbn_sizes, object)
        sizes = [n, dbn_sizes]
        n_rbm = len(sizes)
        initfunct = ""
        rbm = [None]

        class Rbm:
            def __init__(self, cdn=None):
                self.cdn = cdn

        #% create weight initialization function
        # if opts.init_type == []: #'empty' #isinstance(opts.init_type, empty)
        #     initfunct = opts.init_type
        if opts.init_type.isalpha():
            if opts.init_type == 'gauss':
                initfunct = lambda m,n: np.random.normal(0, 0.1, [m,n])
            elif opts.init_type == 'crbm':
                initfunct = crbm_init_weights(int, int)
            # else:
            #     #raise SystemExit("init_type should be either gauss or cRBM")
            #     raise ValueError("init_type should be either gauss or cRBM")
        else:
            raise ValueError("init_type should be either gauss or cRBM")


        @property
        def myfunc(self):
            return self

    for u in range((n_rbm_1)-1):
        rbmList = []
        rbmList.append(Dbn.Rbm)
        Dbn.Rbm.cdn = opts.cdn
        print(Dbn.Rbm.cdn)
        print(rbmList[0].cdn)



    # check cdn if its a function handle use it otherwise create a function from the scalar given

    dbn = Dbn
    return dbn, dbn_sizes                                  #TODO: pass sizes to dbn class better

