from dbncreateopts import *
import dbncreateopts
from dbncreateopts import dbncreateopts
#from dbncreateopts import opts

dbncreateopts()
opts = dbncreateopts()

opts.t_momentum = 0.01
print(opts.momentum(opts.t_momentum, opts.p_i, opts.T, opts.p_f))
#print(opts.momentum_value)
print(opts.learningrate(opts.t_learningrate, opts.eps, opts.f, opts.momentum(opts.t_momentum, opts.p_i, opts.T, opts.p_f)))
print(opts.t_momentum)

#opts = dbncreateopts().opts

#print((opts.cdn))
