import numpy as np
import scipy.io as sio
from dbncreateopts import dbncreateopts
import dbncheckopts
from dbncheckopts import dbncheckopts
from dbnsetup import dbnsetup
from dbntrain import dbntrain
from rbmsemisuplearn import rbmsemisuplearn

# biomag_labeled_1 = sio.loadmat(r"C:\Users\Pap Gergő\PycharmProjects\rbm-tf\data_labeled_for_py_1_2.mat")


biomag_labeled_1 = sio.loadmat("./data/data_labeled_6.mat")
biomag_unlabeled_1 = sio.loadmat("./data/unlabeled_data_5000_set_001.mat")

sizes = [50]

opts, valid_fields = dbncreateopts()
# print("opts: ", opts)

opts.batchsize = 100
# print(sio.whosmat(r"D:\python_project\wip\data_labeled_for_py_1_2.mat"))
print(sio.whosmat("./data/data_labeled_6.mat"))
opts.y_train = biomag_labeled_1["y_train"]
opts.x_train = biomag_labeled_1["x_train"]
opts.x_val = biomag_labeled_1["x_val"]
opts.x_semisup = biomag_unlabeled_1["unlabeled_data_5000_set_001"]
x_train = biomag_labeled_1["x_train"]
# print("opts_ytrain: ",opts.y_train.shape)
# print("opts_learningrate: ",opts.learningrate(opts.t_learningrate, opts.eps, /
#  opts.f, opts.momentum(opts.t_momentum, opts.p_i, opts.T, opts.p_f)))

# ----------dbncheckopts
# print(opts.__dict__.keys())
print("\n")
# fieldnames = opts.__dict__.keys()
# print("fieldnames", fieldnames)

print("\n")



#print("vars opts: \n", vars(opts))

# ---------!dbncheckopts


dbncheckopts(opts,valid_fields)

# print("Sizes sizes: ", sizes)

# opts.t_learningrate = [0.1,0.5]
# print("learningrate multiple elements: ", opts.t_learningrate)
opts.train_function = rbmsemisuplearn                                       # todo : 'train_func' correction in opts

#dbnsetup(sizes,x_train,opts)
#rbmlist = []
rbmlist, dbn, dbn_sizes = dbnsetup(sizes, x_train, opts)
#print("rbmlist[u] at example_1: ", rbmlist[0])

dbntrain(rbmlist[:], dbn, x_train, opts)
# Dbn = dbnsetup(sizes, x_train, opts)

#                      Dbn instance print----------------------------------------------------------------------
# print(Dbn)
# print(dir(Dbn))
# print("Dbn instance sizes: ",Dbn.sizes)
# print("\n Dbn instance initialization function: ", Dbn.initfunct)
# initfunct_test1 = Dbn.initfunct(4,5)
# print(initfunct_test1)

# print(opts.learningrate(eps, f, t_learningrate, momentum_value))
# print(opts.momentum(0.01))
# print(opts.learningrate())

# Dbn.sizes = [500,40]
# print(Dbn.sizes)
# print(Dbn)
