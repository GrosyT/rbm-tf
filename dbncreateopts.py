#def dbncreateopts():

# %DBNCREATEOPTS creates a valid Opts struct
# % The OPTS struct
# %    The following fields are valid
# %
# %         traintype : CD for contrastive divergence, PCD for persistent
# %                     contrastive divergence. see [3,4]
# %               cdn : integer. Number of gibbs steps.
# %                     Applies to both CD and PCD setting. see [3,4]
# %         numepochs : number of epochs
# %         batchsize : minibatch size. mod(n_samples,batchsize) must be 0
# %      learningrate : a function taking current epoch and current momentum as
# %                     as variables and returns a learning rate e.g.
# %                     @(epoch,momentum) 0.1*0.9.^epoch*(1-momentum)
# %          momentum : a function that takes epoch number as input and returns a
# %                     momentum rate e.g.
# %                     T = 50;       % momentum ramp up
# %                     p_f = 0.9;    % final momentum
# %                     p_i = 0.5;    % initial momentum
# %                     @(epoch)ifelse(epoch<T,p_i*(1-epoch/T)+(epoch/T)*p_f,p_f)
# %                L1 : double specifying L1 weight decay
# %                L2 : double specifying L2 weight decay
# %            L2norm : double specifying constraint on the incoming weight sizes
# %                     to each nuron. If the L2norm is above this value the
# %                     weights for this neuron is rescaled to L2norm. See [2]
# %          sparsity ; Use a simple sparsity measure. substract sparsity from the
# %                     hidden biases after each update. see [1]
# %         classRBM  : If this field exists and is 1 then train the DBN where the
# %                     visible layer of the last RBM has the training labels
# %                     added. See "To recognize shapes, first learn to generate
# %                     images" Requires y_train to be spcified.
# %    test_interval  : how often the performance should be measured
# %           y_train : Must be specified if classRBM is 1
# %             x_val : If specified the energy ratio between a training set the
# %                     and the validation set will be caluclated every
# %                     ratio_interval epoch
# %             y_val : if classRBM is a field and x_val is a field this field
# %                     must be specified
# %         x_semisup : unsupervised training examples. For use when the training
# %                     function is @rbmsemisuplearn
# %    early_stopping : Use earlystopping
# %          patience : Patience when using early stopping. Notice that
# %                     epochs that will pass before we stop are
# %                     patience * test_interval. E.g i you want a patience of
# %                     5 epocs and the test_interval is 5 set patience to 1
# %        train_func : @rbmgenerative: Generative rbm training with or without
# %                     labels.
# %                     @rbmdiscriminative: discriminative training. Requires
# %                     training labels.
# %                     @rbmhybrid mix of generative and discriminative, see [1]
# %                     @rbmsemisublearn use unsupervised training. See [1] sec 8
# %                     requires x_unsup to be set. control importance of
# %                     unsupervised training with the beta param
# %                     The semi_sup_type param determines if semisupervised
# %                     training is combined with hybrid, generative or
# %                     discriminative training.
# %
# %          err_func : A function which return a error measure. This applies only
# %                     to a classRBM. The error function
# %                     takes the predicted probabilites as first argument and the
# %                     one-of-K encoded true labels as second arguments. see
# %                     accuracy.m in utils folder.
# %
# %      hybrid_alpha : weigthing of generative and hybrid training objective see
# %                     [1]
# %      semisup_beta : importance of unupservised samples in semi-supervised
# %                     learning.
# %      semisup_type : either @rbmhybrid, @rbmgenerative or @rbmdiscriminative
# %                     see train_func for description.
# %    dropout_hidden : dropout fraction of hidden units.
# %         init_type : initialization of weightes.
# %                     'gauss' init at gaussian with 0 mean and 0.01 std
# %                     'cRBM' init as larochelle in [1] i.e
# %                     weights = randnd(size(weights))-0.5 ./ max(size(weights)).
# %                     Bias units are always initialized at zero.
# %           outfile : after each epoch the best_rbm or rbm is saved to this file


# /% DEFAULT SETTINGS             # https://stackoverflow.com/questions/8948777/create-an-object-without-calling-a-class
def dbncreateopts():           # !!!!!tab
    class Opts:
        traintype = 'CD'
        numepochs = 100
        batchsize = 100
        cdn = 1
        T = 50  # momentum ramp up
        p_f = 0.9  # final momentum
        p_i = 0.5  # initial momentum
        eps = 0.01  # initial learning rate
        f = 0.9  # learning rate decay
        t_learningrate = [0.1]
        t_momentum = [0.01]

        learningrate_lambda = lambda t, momentum: opts.eps * opts.f ** t * (1 - momentum)
        momentum_lambda = lambda t: opts.p_i * (1 - t / opts.T) + (t / opts.T) if t < opts.T else opts.p_f

        def momentum_func(self,t):
            momentum = opts.p_i * (1 - t / opts.T) + (t / opts.T) if t < opts.T else opts.p_f
            return momentum

        def learningrate_func(self, t, momentum):
            learningrate = opts.eps * opts.f ** t * (1 - momentum)
            return learningrate

        L1 = 0.00
        L2 = 0
        L2norm = 0
        sparsity = 0
        classRBM = 1    #default matlab value = 0 and declared in example run code to be value 1
        test_interval = 5


            # a = []
            # for x in y:
            #     a.append(x)
            # a = np.array(a)

            #a = np.array([x for x in y]); or just a = np.array(list(y))

        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_semisup = []
        early_stopping = 0
        patience = 5
        train_function = "rbmgenerative"                    # other options: "rbmdiscriminative","semisup"
        hybrid_alpha = 0.5
        semisup_beta = 0.1
        semisup_type = "rbmhybrid"
        err_func = "accuracy"
        dropout_hidden = 0
        init_type = 'gauss'
        outfile = []

        @property
        def myfunc(self):
            return self

    #missing : valid_fields = fieldnames(opts);

    opts = Opts()                                         # o: opts = Opts   - not instance but the class object
    valid_fields = dir(opts)

    #print(opts)
    #print(dir(Opts))
    #print("valid_fields: \n" ,valid_fields)
    #print("Opts structure info out: ", Opts.test_interval)
    return opts, valid_fields
    #return valid_fields


#opts, valid_fields = dbncreateopts()
#opts = dbncreateopts()
#print(opts)
#print("hello2", opts.test_interval)

#print("valid_fields: \n", valid_fields)

   # learningrate_lambda = lambda t, momentum: opts.eps*opts.f^(t*(1-momentum))
        # momentum_lambda = lambda t: opts.p_i * (1 - t / opts.T) + (t / opts.T) if t < opts.T else opts.p_f
        #
        # def momentum_func(self,t):
        #     momentum = opts.p_i * (1 - t / opts.T) + (t / opts.T) if t < opts.T else opts.p_f
        #     return momentum
        #
        # def learningrate_func(self,t,momentum):
        #     learningrate = opts.eps*opts.f^(t*(1-momentum))
        #     return learningrate
        # Opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);

        # @staticmethod
        # def momentum(t_momentum=None):
        #     if t_momentum is None:
        #         t_momentum = Opts.t_momentum
        #         # print("true")
        #     # else:
        #     #     t_momentum = args
        #     #    print("false")
        #     #t_momentum = Opts.t_momentum
        #     T = 50  # momentum ramp up
        #     p_f = 0.9  # final momentum
        #     p_i = 0.5  # initial momentum
        #     momentum_value = p_i * (1 - t_momentum / T) + (t_momentum / T)*p_f if t_momentum < T else p_f
        #     return momentum_value
        # #momentum_value = momentum(t_momentum, p_i, T, p_f)
        # #momentum_value = momentum()
        #
        # def learningrate(t_learningrate=None):
        #     if t_learningrate is None:
        #         t_learningrate = Opts.t_learningrate
        #     eps = Opts.eps
        #     f = Opts.f
        #     momentum_value = Opts.momentum()#Opts.momentum_value
        #     if len(t_learningrate) == 1:
        #         learning_rate_value = eps*f**(t_learningrate[0]*(1-momentum_value))
        #     else:
        #         for i in t_learningrate:
        #             learning_rate_value = eps * f ** (t_learningrate[i] * (1 - momentum_value))
        #     return learning_rate_value
        #momentum = lambda t, p_i, T, p_f: p_i*(1-t/T)+(t/T) if t < T else p_f