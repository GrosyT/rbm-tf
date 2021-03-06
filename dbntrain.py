from rbmtrain import rbmtrain
import sys
sys.path.insert(0, './util/')
from sigm import sigm

# function dbn = dbntrain(dbn, x_train, opts)
# %%DBNTRAIN train a DBN by stacking RBM's
# % see dbncreateopts for a description of the opts struct. Use dbnsetup
# % to create the dbn struct.
# %
# % EXAMPLE
# %  sizes = [200];   % hidden layer size
# %  [opts, valid_fields] = dbncreateopts();
# %  dbncheckopts(opts,valid_fields);
# %  disp(opts)
# %  dbn = dbnsetup(sizes, train_x, opts);
# %  dbn = dbntrain(dbn, train_x, opts);
# %
# % See also DBNCREATEOPTS DBNCHECKOPTS DBNSETUP DBNTRAIN
# %
# % Copyright june 2014 by Sřren Sřnderby


def dbntrain(rbmlist, dbn, x_train, opts):
    training = "rbmtrain"  # call rbmtrain.m func. handle

    n_rbm = len(rbmlist)  # print(n_rbm)
    #
    # print("rbmlist at dbntrain : ", rbmlist)
    # print("rbmlist[u] at dbntrain : ", rbmlist[0])
    # print("rbmlist[u].cdn at dbntrain : ", rbmlist[0].cdn)
    aline = "--------------------------------------------------------------------------------"
    print("\n",aline,"\n","Training RBM 1\n",aline)

    # rbmlist[0] = rbmtrain(rbmlist[0], x_train, opts)

    # matlab-szerű ötlet, külön kimenti az rbm-struktúrát. Abból is az elsőre külön meghívva a training-et.
    rbm = rbmlist[0]
    rbm = rbmtrain(rbm, x_train, opts) # :o rbmlist = rbmtrain(rbm, x_train, opts)
    rbmlist[0] = rbm

    for i in range(1, n_rbm):
        if rbmlist[i].classRBM == 1 and n_rbm == i:
            ye = opts.y_train
        else:
            ye = []

        print("\n",aline,"\n","Training RBM ", i, "\n", aline,)
        x_train = rbmlist[i-1].rbmup(rbmlist[i-1], x_train, ye, sigm)

        if opts.x_val.any():
            opts.x_val = rbmlist[i-1].rbmup(rbmlist[i-1], opts.x_val, ye, sigm)

        rbm = rbmlist[i]
        rbm = training(rbm, x_train, opts)
        rbmlist[i] = rbm

    return rbmlist





    # rbmtrain - dbntrain v1.
    #  eredeti ötletem, magát az rbm-eket tároló listát adja át a trainingnek
    # rbmlist = rbmtrain(rbmlist, x_train, opts)



    # rbmlist[0].d = 5
    # print(rbmlist[0].d)
    # print("rbmlist[0].cdn at dbntrain: ",rbmlist[0].cdn)
