from dbncreateopts import dbncreateopts
import pprint

#DBNCHECKOPTS checks the validity of the opts struct
#
# see also, DBNSETUP, DBNTRAIN, DBNCREATEOPTS
#
# Copyright Sřren Sřnderby july 2014


#-----------------------------test
#opts, valid_fields = dbncreateopts()


def dbncheckopts(opts, valid_fields):
    fields = dir(opts)

    #print(fields, sep="\n")
    #print("\n".join(fields))                             #-print each element in new line

    sorted_fields = sorted(fields)
    #print("Sorted fields: \n", sorted(fields))
    #pprint(sorted_fields)

    # fields.sort()
    # print("fields.sort: \n", fields)
    #sorted_fields[1] = "L11"                              #---test fields equality

    fields_bool = sorted_fields == valid_fields
    #print(fields_bool)

    try:
        content = fields_bool
        if not content:
            raise ValueError('Opts fields contain unallowed element')
    except (ValueError, IndexError):
        exit('Could not complete request, because Opts struct contains wrong element(s)') #exit('Could not complete request')


    #valid = @(f) isfield(opts,f) == 1 && ~isempty(opts.(f));

    # valid =
    #
    # function_handle
    # with value:
    #
    #     @(f)
    #
    #     isfield(opts, f) == 1 & & ~isempty(opts.(f))

    #% check if y is given if class rbm + check y size if x is given

    if opts.classRBM == 1:
        if len(opts.y_train) == 0 or opts.y_train is None:  # o: if len(opts.y_train) == 0 or opts.y_train == ""
            raise ValueError('Y train can not be empty in case of classRBM!')
    if opts.train_function == "rbmsemisuplearn" and opts.classRBM != 1:
        raise ValueError('Semisupervised training without labels does not make sense, use RBMGENERATIVE')
    if opts.train_function != 'rbmgenerative' and opts.train_function != "rbmdiscriminative":
        print("TRAINING FUNCTION: ", opts.train_function)
        #raise ValueError('Training function not recognised!')

    print("DBN CHECK OPTS: Done!\n")





#dbncheckopts(Opts)
#dbncheckopts(opts, valid_fields)

