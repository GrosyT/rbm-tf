import numpy as np
from numpy import genfromtxt
import scipy.io as sio
from dbncreateopts import dbncreateopts
import dbncheckopts
from dbncheckopts import dbncheckopts
from dbnsetup import dbnsetup
from dbntrain import dbntrain
from rbmsemisuplearn import rbmsemisuplearn
from rbmdiscriminative import rbmdiscriminative
from rbmgenerative import rbmgenerative
from dbnpredict import dbnpredict
import h5py


# biomag_labeled_1 = sio.loadmat(r"C:\Users\Pap Gergő\PycharmProjects\rbm-tf\data_labeled_for_py_1_2.mat")


biomag_labeled_1 = sio.loadmat("./data/data_labeled_6.mat")
biomag_unlabeled_1 = sio.loadmat("./data/unlabeled_data_5000_set_001.mat")
# biomag_unlabeled_1 = []
# with h5py.File('./data/Biomag_unlabeled_3.mat', 'r') as f:
#     biomag_unlabeled_1 = np.array(f["unlabeled_min_max_norm"])
#     print(biomag_unlabeled_1.shape)
#biomag_unlabeled_1 = f["unlabeled_min_max_norm"]
#biomag_unlabeled_2 = sio.loadmat("./data/Biomag_unlabeled_3.mat")
#biomag_unlabeled_2 = genfromtxt(r"I:\OneDrive - Szegedi Tudományegyetem\egyetem\tdk\program\unlabeled_mmn.mat", delimiter=',')

sizes = [500]

opts, valid_fields = dbncreateopts()
# print("opts: ", opts)
opts.numepochs = 50
opts.patience = 15
opts.batchsize = 10

opts.train_function = rbmgenerative                         # todo : 'train_func' correction in opts
opts.semisup_type = rbmdiscriminative

opts.learningrate = 0.05
opts.momentum = 0.001

opts.semisup_beta = 0.1
opts.traintype = "CD"
opts.init_type = "crbm"


# print(sio.whosmat(r"D:\python_project\wip\data_labeled_for_py_1_2.mat"))
print(sio.whosmat("./data/data_labeled_6.mat"))
opts.y_train = biomag_labeled_1["y_train"]
opts.x_train = biomag_labeled_1["x_train"]
opts.x_val = biomag_labeled_1["x_val"]
opts.y_val = biomag_labeled_1["y_val"]
opts.x_semisup = biomag_unlabeled_1["unlabeled_data_5000_set_001"]
#print(biomag_unlabeled_1.shape) #.o["unlabeled_data_5000_set_001"]
x_train = biomag_labeled_1["x_train"]
x_test = biomag_labeled_1["test_x"]
y_test = biomag_labeled_1["test_y"]

# ----------dbncheckopts
# print(opts.__dict__.keys())
print("\n")
# print("fieldnames", fieldnames)

print("\n")

# ---------!dbncheckopts


dbncheckopts(opts,valid_fields)

# print("Sizes sizes: ", sizes)

#dbnsetup(sizes,x_train,opts)
#rbmlist = []
rbmlist, dbn, dbn_sizes = dbnsetup(sizes, x_train, opts)
#print("rbmlist[u] at example_1: ", rbmlist[0])

dbn = dbntrain(rbmlist[:], dbn, x_train, opts)
# Dbn = dbnsetup(sizes, x_train, opts)


pred_y = dbnpredict(dbn, x_test)
pred_y = pred_y + 1 # data labels encoded differently from python 0 starting index
result = pred_y == np.reshape(y_test,(y_test.shape[0]))
accuracy_final = np.sum(result) / x_test.shape[0] * 100
print("Accuracy on test: ",accuracy_final,"%")


np.savetxt('RBM_accuracy.txt', pred_y, fmt='%d')
