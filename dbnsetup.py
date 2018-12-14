# %%DBNSETUP creates a propr dbn struct
# %     INPUT
# %        sizes : A vector with hidden layer sizes
# %            x : used to specify size of first hidden layer
# %         opts : a struct with options see dbncreateopts
import numpy as np
import types
from rbmupclassrbm import rbmupclassrbm
from rbmdownyclassrbm import rbmdownyclassrbm


# create weight initialization function
def init_weights(m, n, opts):  # todo complete weight init func in cRBM case
    if opts.init_type == "gauss":
        # initfunc = lambda m, n : np.random.normal(0, 0.1,(m, n))
        initfunc = np.random.normal(0, 0.01, (m, n))
        return initfunc
    elif opts.init_type == "crbm":
        # initialize weights from uniform distribution. As described in
        # Learning Algorithms for the Classification Restricted Boltzmann machine
        m_max = max(m, n)
        interval_max = m_max ** (-0.5)
        interval_min = -interval_max
        weights = interval_min + (interval_max - interval_min) * np.random.uniform(0, 1, (m, n))
        assert np.amax(weights) <= interval_max
        assert np.amin(weights) >= interval_min
        initfunc = weights
        return initfunc
    else:
        raise ValueError("init_type should be either gauss or cRBM")

    # check cdn if its a function handle use it otherwise create a function from the scalar given


def dbnsetup(sizes, x_train, opts):
    n = x_train.shape[1]  # [2094,254] -> n = 254
    dbn_sizes = [n] + sizes  # dbn_sizes = [254,50]
    n_rbm_1 = len(dbn_sizes) - 1  # n_rbm_1 = 1

    # # test init weights
    # test_init_weights = init_weights(50, 40, opts)
    # print("test init weights: ", test_init_weights[4,5])

    class Dbn:
        assert isinstance(dbn_sizes, object)
        sizes = [n, dbn_sizes]
        n_rbm = len(sizes)
        # initfunc = ""
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
            d = None
            vd = None
            W = None
            vW = None
            b = None
            vb = None
            c = None
            vc = None
            rand = None
            zeros = None
            rbmdowny = None
            rbmup = None

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
                self.d = None
                self.vd = None
                self.W = None
                self.vW = None
                self.b = None
                self.vb = None
                self.c = None
                self.vc = None
                self.rand = None
                self.zeros = None
                self.rbmdowny = None
                self.rbmup = None

            def create_func(self, val):                         # todo: possible bug with matlab function handle
                # takes a scalar val or function handle and returns a function returning val if val is not a function
                if isinstance(val, types.FunctionType):
                    ret = val
                else:
                    ret = lambda epoch: val
                return ret

        # % create weight initialization function
        # if opts.init_type == []: #'empty' #isinstance(opts.init_type, empty)
        #     initfunct = opts.init_type
        # if opts.init_type.isalpha():
        #     if opts.init_type == 'gauss':
        #         initfunct = lambda m, n: np.random.normal(0, 0.1, [m,n])
        #     elif opts.init_type == 'crbm':
        #         initfunct = crbm_init_weights(int, int)
        #     # else:
        #     #     #raise SystemExit("init_type should be either gauss or cRBM")
        #     #     raise ValueError("init_type should be either gauss or cRBM")
        # else:
        #     raise ValueError("init_type should be either gauss or cRBM")

        @property
        def myfunc(self):
            return self

    rbmlist = []  # store multiple rbm-s
    for u in range(n_rbm_1):

        rbmlist.append(Dbn.Rbm())
        # Dbn.Rbm.cdn = opts.cdn
        rbmlist[u].cdn = Dbn.Rbm.create_func(Dbn.Rbm.create_func, opts.cdn)
        # rbmlist[u].cdn = create_func(opts)   not sure about# t0d0: create function cdn matlab dbnsetup.m line 41 | 168
        # print("list 0 index: ", rbmlist[0].cdn)
        # rbmlist.append(Dbn.Rbm)
        # rbmlist[1].cdn = opts.cdn
        # print("list 1 index: ", rbmlist[1].cdn)

        # if one learningrate/momentum function use this for all
        # otherwise use individual learningrate/momentum for each rbm
        if len(opts.t_learningrate) == n_rbm_1 and n_rbm_1 != 1:
            rbmlist[u].learningrate = opts.t_learningrate[u]
        elif len(opts.t_learningrate) == 1:

            rbmlist[u].learningrate = opts.learningrate_func                                     # -o: rbmlist[u].learningrate = opts.t_learningrate
            # raise ValueError("learnfunc. should be 1 or nrbm")
        else:
            assert len(opts.t_learningrate) == 1, "learnfunc. should be 1 or nrbm"

        if len(opts.t_momentum) == n_rbm_1 and n_rbm_1 != 1:
            rbmlist[u].learningrate = opts.t_learningrate[u]
        elif len(opts.t_momentum) == 1:
            rbmlist[u].momentum = opts.momentum_func  # -o: rbmlist[u].momentum = opts.t_momentum
            # raise ValueError("Momentum func. should be 1 or nrbm")
        else:
            assert len(opts.t_momentum) == 1, "Momentum func. should be 1 or nrbm"

        # regularization parameters
        rbmlist[u].L2 = opts.L2
        rbmlist[u].L1 = opts.L1
        rbmlist[u].L2norm = opts.L2norm
        rbmlist[u].sparsity = opts.sparsity
        rbmlist[u].dropout_hidden = opts.dropout_hidden

        # error stuff
        rbmlist[u].err_func = opts.err_func
        rbmlist[u].error = []
        rbmlist[u].val_error = []
        rbmlist[u].train_error = []
        rbmlist[u].train_error_measures = []
        rbmlist[u].val_error_measures = []
        rbmlist[u].energy_ratio = []

        # early stopping for non top layers not implemented, because they are not classRBMS
        if (n_rbm_1 - 1) == u:
            rbmlist[u].early_stopping = opts.early_stopping
        else:
            rbmlist[u].early_stopping = 0
        rbmlist[u].patience = opts.patience

        vis_size = dbn_sizes[u]
        hid_size = dbn_sizes[u + 1]

        if opts.classRBM == 1 and u == n_rbm_1 - 1:  # init bias and weights for class vectors
            rbmlist[u].classRBM = 1
            rbmlist[u].train_func = opts.train_function
            n_classes = opts.y_train.shape[1]
            # o: n_classes = np.amax(np.transpose(opts.y_train)).astype(int)
            # done? to-do: modify to accomodate other dimensions current: one-hot
            # n_classes = n_classes.astype(int)
            rbmlist[u].U = init_weights(hid_size, n_classes, opts)  # (hidden_size, n_classes)
            rbmlist[u].vU = np.zeros((hid_size, n_classes))
            rbmlist[u].d = np.zeros((n_classes, 1))
            rbmlist[u].vd = np.zeros((n_classes, 1))

        else:  # for non-top layers use generative training
            rbmlist[u].classRBM = 0
            rbmlist[u].train_func = rbmgenerative
            rbmlist[u].U = []
            rbmlist[u].vU = []
            rbmlist[u].d = []
            rbmlist[u].vd = []

        rbmlist[u].W = init_weights(hid_size, vis_size, opts)
        rbmlist[u].vW = np.zeros((hid_size, vis_size))
        rbmlist[u].b = np.zeros((vis_size, 1))
        rbmlist[u].vb = np.zeros((vis_size, 1))
        rbmlist[u].c = np.zeros((hid_size, 1))
        rbmlist[u].vc = np.zeros((hid_size, 1))

        # #test rbmlist.bias - zeros
        #print("rbmlist[u].b: ",rbmlist[u].b)

        # for non class RBM's rbmy should return empty. To avoid if statement
        # create a function returning empty otherwise use rbmdowny

        rbmlist[u].rand = np.random.rand  # todo: replicate matlab function handle calls better
        rbmlist[u].zeros = lambda zeros: np.zeros(0)

        if rbmlist[u].classRBM:  # todo: implement these as separate functions (based on matlab func. handle or .m file)
            rbmlist[u].rbmdowny = rbmdownyclassrbm
            rbmlist[u].rbmup = rbmupclassrbm
        else:
            rbmlist[u].rbmdowny = "rbmdownynotclass"
            rbmlist[u].rbmup = "rbmdupnotclassrbm"

        # if len(opts.t_learningrate) == 0:

        # if rbmlist[u].learningrate == 1:

        # if len(opts.learningrate())
    #print("rbmlist at dbnsetup:", rbmlist)
    #print("rbmlist[u] at dbnsetup:", rbmlist[0])

    dbn = Dbn()
    # print("dbn = Dbn at dbnsetup: ",dbn)
    return rbmlist[:], dbn, dbn_sizes  # TODO: pass sizes to dbn class better
