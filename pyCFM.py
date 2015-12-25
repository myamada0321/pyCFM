import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator


class CFM:
    """ Convex Factorization Machine with Hazan's algorithm (squared loss)
        Yamada, M. et. al., http://arxiv.org/abs/1507.01073


    Model:

    f(x;w,W) = w^t z + 0.5 trace(W (xx^t - diag(x.*x)))

    Parameters
    ----------

    num_iter: int
       The number of iteration for Hazan's algorithm: (default: 100)

    reg_W: double
       The regularization parameter for interaction term (2-way factors) (default 100)

    w: double
       d + 1 dimensional vector the global bias (w[0]) and user-item bias (w[1:])

    U: double
       d by T dimensional matrix such that W = U U^t

    """

    def __init__(self, num_iter=100, reg_W = 100):
        self.num_iter = num_iter
        self.reg_W = reg_W

    def fit(self,X,Y):

        if type(Y) != np.ndarray:
            Y = np.array(Y)


        [d,n] = X.shape

        #Add bias for training data
        Z = sp.vstack((np.ones(n),X),format='csr')

        T = self.num_iter
        eta = self.reg_W

        self.U = np.zeros((d,T))
        P = np.zeros((d,T))
        lam = np.ones(T)
        w = np.zeros((d+1,1))
        fval = np.zeros(T)



        #matvec function for eig
        def mv_eig(a):
            return X.dot(X.transpose().dot(a)*tr_err)

        eigcalc = LinearOperator((d,d),matvec=mv_eig,dtype='float64')

        #matvec function for cg
        def mv_cg(a):
            return Z.dot(Z.transpose().dot(a))

        cgcalc = LinearOperator((d+1,d+1),matvec=mv_cg,dtype='float64')

        #Preconditioner
        M = sp.diags(np.array(1/(Z.sum(1)+0.00000001)).flatten(),0)

        for t in range(0,T):

            if t != 0:
                tmp = X.transpose().dot(self.U[:,0:t])
                fQ = 0.5*((tmp*tmp).sum(1) - X.multiply(X).transpose().dot(np.sum(self.U[:,0:t]*self.U[:,0:t],1)))
            else:
                fQ = np.zeros(n)

            ZY = Z.dot(Y-fQ)

            #Conjugate Gradient: Solve Zw = Y
            wout = sp.linalg.cg(cgcalc,ZY,tol=1e-06,maxiter=1000,x0=w,M=M)
            self.w = wout[0]

            tr_err = Y - Z.transpose().dot(self.w) - fQ

            #Frank-Wolfe update: eigs(X diag(tr_err) X^t, 1)
            [l,pout] = sp.linalg.eigsh(eigcalc,k=1,which='LA',maxiter=1000, tol=1e-1)
            p = np.real(pout).flatten()

            #Optimal step size
            err = eta*((X.transpose().dot(p))**2 - (X.multiply(X)).transpose().dot(p*p)) - fQ
            alpha = np.dot(tr_err,err)/np.dot(err,err)

            #Update
            P[:,t] = np.sqrt(eta)*p
            lam[0:t] = lam[0:t] - alpha*lam[0:t]
            lam[t] = max(1e-10,alpha)

            sqlam = np.sqrt(lam)
            self.U[:,0:(t+1)] = P[:,0:(t+1)]*sqlam[0:(t+1)]

            #Training RMSE
            fval[t] = np.sqrt(np.mean(tr_err**2))

            #fval: training RMSE
            print 'Training RMSE: %f ' % fval[t]

    def predict(self,Xte):
        #Compute fQ (2-way factor)
        tmp = Xte.transpose().dot(self.U)
        fQ = 0.5*((tmp*tmp).sum(1) - Xte.multiply(Xte).transpose().dot(np.sum(self.U*self.U,1)))

        Y_hat = self.w[0] + Xte.transpose().dot(self.w[1:]) + fQ

        return Y_hat