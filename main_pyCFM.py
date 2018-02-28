
import numpy as np
import scipy.sparse as sp
from pyCFM import CFM


if __name__ == '__main__':
    #Movielens 1M data
    #Get ml-1m data from http://grouplens.org/datasets/movielens/

    fin = open('ml-1m/ratings.dat','r')

    #Transform user-rating data for FM format
    row_ind  = []
    col_ind = []
    Y = []
    rmax = 0
    for line in fin:
        data_in = line.strip().split('::')
        rind = int(data_in[0]) -1
        cind = int(data_in[1]) -1
        #X_dict[rind,cind] = int(data[2])

        row_ind.append(rind)
        col_ind.append(cind)
        Y.append(int(data_in[2]))

        if rmax < rind:
            rmax = rind

    fin.close()

    row_ind = np.array(row_ind)
    col_ind = np.array(col_ind) + rmax + 1
    ind1 = np.concatenate((row_ind,col_ind))
    ind2 = np.concatenate((range(len(Y)), range(len(Y))))
    X = sp.csr_matrix((np.ones(len(Y)*2),(ind1,ind2)))

    np.random.seed(1)
    randind = np.random.permutation(len(Y))

    #80% of data for training and 20% of data for test
    ntr = int(len(Y)*0.8)

    Y = np.array(Y)

    #Training set
    Xtrain = X[:,randind[0:ntr]]
    Ytrain = Y[randind[0:ntr]]

    #Test set
    Xtest = X[:,randind[(ntr+1):]]
    Ytest = Y[randind[(ntr+1):]]

    #CFM training and test

    #Model
    cfm_model = CFM(num_iter=100, reg_W = 4000)

    #Training CFM model using Hazan's algorithm
    cfm_model.fit(Xtrain,Ytrain)

    #Prediction
    Ytest_hat = cfm_model.predict(Xtest)

    #Test RMSE
    rmse = np.sqrt(np.mean((Ytest - Ytest_hat)**2))

    print('Test RMSE: %f' % rmse)