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
    n_rbm_1 = len(dbn_sizes) - 1  # n_rbm_1 = 1

    class Dbn:
        assert isinstance(dbn_sizes, object)
        sizes = [n, dbn_sizes]
        n_rbm = len(sizes)
        initfunct = ""
        rbm = [None]

        class Rbm:
            cdn = None
            learningrate = None
            momentum = None
            L2 = None
            L1 = None
            L2norm = None
            sparsity = None
            dropout_hidden = None
            err_func = None
            error = None
            val_error = None
            train_error = None
            train_error_measures = None
            val_error_measures = None
            energy_ratio = None
            patience = None
            early_stopping = None
            classRBM = None
            train_func = None
            U = None
            vU = None

            def __init__(self):
                self.cdn = None
                self.learningrate = None
                self.momentum = None
                self.L2 = None
                self.L1 = None
                self.L2norm = None
                self.sparsity = None
                self.dropout_hidden = None
                self.err_func = None
                self.error = None
                self.val_error = None
                self.train_error = None
                self.train_error_measures = None
                self.val_error_measures = None
                self.energy_ratio = None
                self.patience = None
                self.early_stopping = None
                self.classRBM = None
                self.train_func = None
                self.U = None
                self.vU = None

        #% create weight initialization function
        # if opts.init_type == []: #'empty' #isinstance(opts.init_type, empty)
        #     initfunct = opts.init_type
        if opts.init_type.isalpha():
            if opts.init_type == 'gauss':
                initfunct = lambda m, n: np.random.normal(0, 0.1, [m,n])
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
        rbmlist[u].cdn = opts.cdn
        #print("list 0 index: ", rbmlist[0].cdn)
        # rbmlist.append(Dbn.Rbm)
        # rbmlist[1].cdn = opts.cdn
        ##print("list 1 index: ", rbmlist[1].cdn)
        if len(opts.t_learningrate) == n_rbm_1 & n_rbm_1 != 1:
            rbmlist[u].learningrate = opts.t_learningrate[u]
        elif (len(opts.t_learningrate) == 1) != 1:
            rbmlist[u].learningrate = opts.t_learningrate
            raise ValueError("learnfunc. should be 1 or nrbm")


        if len(opts.t_momentum) == n_rbm_1 and n_rbm_1 != 1:
            rbmlist[u].learningrate = opts.t_learningrate[u]
        elif (len(opts.t_momentum) == 1) != 1:
            rbmlist[u].momentum = opts.t_momentum
            raise ValueError("Momentum func. should be 1 or nrbm")


        #regularization parameters
        rbmlist[u].L2 = opts.L2
        rbmlist[u].L1 = opts.L1
        rbmlist[u].L2norm = opts.L2norm
        rbmlist[u].sparsity = opts.sparsity
        rbmlist[u].dropout_hidden = opts.dropout_hidden

        #error stuff
        rbmlist[u].err_func = opts.err_func
        rbmlist[u].error = []
        rbmlist[u].val_error = []
        rbmlist[u].train_error = []
        rbmlist[u].train_error_measures = []
        rbmlist[u].val_error_measures = []
        rbmlist[u].energy_ratio = []

        #early stopping for non top layers not implemented, because they are not classRBMS
        if (n_rbm_1 - 1) == u:
            rbmlist[u].early_stopping = opts.early_stopping
        else:
            rbmlist[u].early_stopping = 0
        rbmlist[u].patience = opts.patience

        vis_size = dbn_sizes[u]
        hid_size = dbn_sizes[u+1]

        if opts.classRBM == 1 and u == n_rbm_1-1:
            #init bias and weights for class vectors
            rbmlist[u].classRBM = 1
            rbmlist[u].train_func = opts.train_function
            n_classes = max(np.transpose(opts.y_train)) #todo: modify to accomodate other dimensions (current: one-hot)




        #if len(opts.t_learningrate) == 0:

        #if rbmlist[u].learningrate == 1:

        #if len(opts.learningrate())
    print("rbmlist :", rbmlist)

    # check cdn if its a function handle use it otherwise create a function from the scalar given

    dbn = Dbn
    return dbn, dbn_sizes                                  #TODO: pass sizes to dbn class better

