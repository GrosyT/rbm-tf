# %%DBNSETUP creates a propr dbn struct
# %     INPUT
# %        sizes : A vector with hidden layer sizes
# %            x : used to specify size of first hidden layer
# %         opts : a struct with options see dbncreateopts
import numpy as np



def crbm_init_weights(m, n):                    #todo complete weight init func in cRBM case
    pass

def dbnsetup(sizes, x_train, opts):
    n = x_train.shape[1]  #[2094,254] -> n = 254
    dbn_sizes = [n] + sizes  # dbn_sizes = [254,50]
    n_rbm_1 = len(dbn_sizes) -1  # n_rbm_1 = 1

    class Dbn:
        assert isinstance(dbn_sizes, object)
        sizes = [n, dbn_sizes]
        n_rbm = len(sizes)
        initfunct = ""
        rbm = [None]

        class Rbm:
            def __init__(self, cdn=None):
                self.cdn = cdn
                self.learningrate

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

    rbmlist = []                          #store multiple rbm-s
    for u in range(n_rbm_1):

        rbmlist.append(Dbn.Rbm)
        #Dbn.Rbm.cdn = opts.cdn
        rbmlist[0].cdn = opts.cdn
        #print("list 0 index: ", rbmlist[0].cdn)
        # rbmlist.append(Dbn.Rbm)
        # rbmlist[1].cdn = opts.cdn
        ##print("list 1 index: ", rbmlist[1].cdn)
        if len(opts.t_learningrate) == n_rbm_1 & n_rbm_1 != 1:
            rbmlist[u].learningrate = opts.t_learningrate[u]
        elif (len(opts.t_learningrate) == 1) != 1:
            raise ValueError("learnfunc. should be 1 or nrbm")
            rbmlist[u].learningrate = opts.t_learningrate

        #if len(opts.t_learningrate) == 0:

        #if rbmlist[u].learningrate == 1:

        #if len(opts.learningrate())
    print("rbmlist :", rbmlist)

    # check cdn if its a function handle use it otherwise create a function from the scalar given

    dbn = Dbn
    return dbn, dbn_sizes                                  #TODO: pass sizes to dbn class better

