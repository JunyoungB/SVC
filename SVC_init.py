import numpy as np
import time
import scipy.io as sio
import sklearn.metrics.pairwise as kernels
import numpy.linalg as la
from scipy.linalg import inv
from scipy.sparse import csgraph as cg
from IPython import embed
from cvxopt import matrix
from cvxopt import solvers
from sklearn.metrics.pairwise import euclidean_distances
from numpy import matlib as ml
from scipy.optimize import *



def kradius(X, model):
    #
    # % =========================================================
    # %
    # % KRADIUS computes the squared distance between vector in kernel space
    # % and the center of support.
    # %
    # % Implemented by Kyu-Hwan Jung
    # % April 26, 2010.
    # %
    # % * The source code is available under the GNU LESSER GENERAL PUBLIC
    # % LICENSE, version 2.1.
    d = np.zeros([X.shape[1], 1])  ######################
    if model['support_type'] == 'SVDD':
        [dim, num_data] = X.shape
        x2 = diagker(X, model['options']['ker'], model['options']['arg'])
        Ksvx = kernel(input1=X, input2=model['sv']['X'], ker=model['options']['ker'], arg=model['options']['arg'])
        d = x2 - 2 * np.dot(Ksvx, model['Alpha']) + model['b'] * np.ones((num_data, 1))
    else:  # 'GP'
        for i in range(X.shape[1]):
            # %[predict_label, accuracy] = var_gpr(X(:,i)', model.X, model.inv_C, model.hyperparams);
            predict_label = var_gpr(np.reshape(X[:, i], [1, -1]), model['X'], model['inv_C'], model['hyperparams'])
            d[i][:] = np.reshape(predict_label, [1, -1])
    return d

def kernel(input1, ker, arg, input2=None):
    if input2 is None:

        input1 = input1.T
        if ker == 'linear':
            K = kernels.linear_kernel(input1)
            #   polynomial은 미구현. 파라미터가 좀 다른듯...
            #    if ker == 'poly':
            #        K = kernels.polynomial_kernel(input, )

        if ker == 'rbf':
            gamma = (0.5 / (arg * arg))
            K = kernels.rbf_kernel(input1, gamma=gamma)
        if ker == 'sigmoid':
            K = kernels.sigmoid_kernel(input1, gamma=arg[0], coef0=arg[1])

        return K

    else:

        input1 = input1.T
        input2 = input2.T
        if ker == 'linear':
            K = kernels.linear_kernel(input1, input2)
            #   polynomial은 미구현. 파라미터가 좀 다른듯...
            #    if ker == 'poly':
            #        K = kernels.polynomial_kernel(input, )

        if ker == 'rbf':
            gamma = (0.5 / (arg * arg))
            K = kernels.rbf_kernel(input1, input2, gamma=gamma)
        if ker == 'sigmoid':
            K = kernels.sigmoid_kernel(input1, input2, gamma=arg[0], coef0=arg[1])

        return K


def qpssvm(H, f, b, I):
    P = matrix(H, tc='d')
    q = matrix(f.reshape(f.shape[0], 1), tc='d')

    G1 = np.eye(I.shape[0])
    G2 = np.eye(I.shape[0])
    G2 = (-1) * G2
    G = matrix(np.concatenate((G1, G2), axis=0), tc='d')

    h1 = np.repeat(b, I.shape[0])
    h1 = h1.reshape(h1.shape[0], 1)
    h2 = np.repeat(0, I.shape[0])
    h2 = h2.reshape(h2.shape[0], 1)
    h = matrix(np.concatenate((h1, h2), axis=0), tc='d')

    sol = solvers.qp(P, q, G, h)

    x = sol['x']
    x = np.array(x)
    fval = sol['primal objective']
    return [x, fval]


def diagker(X, ker, arg):
    diagK = np.diag(kernel(input1=X, ker=ker, arg=arg))
    diagK = diagK.reshape(diagK.shape[0], 1)
    return diagK


def var_gpr(test, input, inv_C, hyperpara):
    # % Variance in Gaussian Process Regression used as Support Funtion of Clustering
    # %
    # % The variance function of a predictive distribution of GPR
    # % sigma^2(x) = kappa - k'C^(-1)k
    # %==========================================================================
    # % Implemented by H.C. Kim Jan. 16, 2006
    # % Modified by Sujee Lee at September 10, 2014.
    # %
    # % * The source code is available under the GNU LESSER GENERAL PUBLIC
    # % LICENSE, version 2.1.
    # %==========================================================================
    [D, n] = input.shape
    [D, nn] = test.shape
    expX = hyperpara
    a = np.zeros([nn, n])

    for d in range(D):
        a = a + expX[0][d] * (np.tile(input[d, :], [nn, 1]) - np.tile(np.reshape(test[d, :], [-1, 1]), [1, n])) ** 2

    a = expX[1] * np.exp(-0.5 * a)
    b = expX[1]
    mul = a.dot(inv_C.T)
    dmul = np.multiply(a, mul)
    s_a_inv_C_a = np.sum(dmul, axis=1)
    var = b - s_a_inv_C_a

    return var


# coding: utf-8

# In[ ]:



"""
Created on Wed Aug 30 19:49:55 2017

@author: User
"""



###Tutorial
"""
data_name='ring'
input = load_data(data_name)
support = "GP"
hyperparams = [100*np.ones((input.shape[0],1)), 1, 10]
st=time.time()
model = supportmodel(input,support,hyperparams)
et=time.time()
print("Training time:", et-st)
print("---------------------------------")
labmodel = labeling(model,"CG-SC")
labmodel.run()
et1=time.time()
print("Labeling time:", et1-et)
print("---------------------------------")
embed()
"""

##Structure
"""
load_data
class support model
    __init__
    gp_normalize
    gp
    get_inv_C
    svdd_normalize
    svdd(Son)
    kernel(Son)
    qpssvm(Son)
class labeling
    __init__
    run
    findAdjMatrix
    kradius
    cgsc
    smsc(Byun)
    tmsc(?)
    fmsc(?)
    vmsc(?)
    findSEPs(Byun)
    findTPs(Byun)
var_gpr
my_R_GP
"""

def load_data(data_name):
    data=sio.loadmat('data/'+data_name)
    if data=='toy':
        input=data['X']
    else:
        input=data['input'].T
    return input
    
class supportmodel:
    def __init__(self, input, support, supportopt, hyperparams):
        self.input=input  ### input.shape = [dim, N_sample]
        assert type(self.input) is np.ndarray, 'ERROR: input type must be numpy.ndarray'
        self.support=support  ### 'SVDD' or 'GP'
        assert self.support=='SVDD' or self.support=='GP', 'ERROR: Support must be \'SVDD\', or \'GP\''
        if self.support=='SVDD':
            self.svdd_normalize()
            self.svdd_params=supportopt
            self.svdd()
        elif self.support=='GP':
            self.gp_normalize()
            self.gp_params=hyperparams
            assert self.gp_params[0].shape[0]==self.input.shape[0], "ERROR: invalid gp_params shape"
            self.gp()

    def gp_normalize(self):  ##my_normalize2
        #% input [dim x num_data]
        #% output [dim x num_data]

        [dim, num]=self.input.shape
        max_val = np.max(np.max(self.input)) ###############np
        min_val = np.min(np.min(self.input))
        self.normalized_input = (self.input-np.tile(min_val,[dim,num]))/np.tile(max_val-min_val,[dim,num]) 
        
    def gp(self):  ###using var_gpr
        ## Gaussian Process Support Function for Clustering 
        ##
        ##  Gaussian process support function which is the variance function of a
        ##  predictive distribution of GPR : 
        ##   sigma^2(x) = kappa - k'C^(-1)k
        ##  where covariance matrix C(i,j) is a parameterized function of x(i) and
        ##  x(j) with hyperparameters Theta, C(i,j) = C(x(i),x(j);Theta),
        ##  kappa = C(x~,x~;Theta) for a new data point x~ = x(n+1),
        ##  k = [C(x~,x1;Theta),...,C(x~,x(n);Theta)]
        ##
        ## Synopsis:
        ##  model = gp(input)
        ##  model = gp(input,hyperparams)
        ##
        ## Description :
        ## It computes variance function of gaussian process regression learned from
        ## a training data which can be an estimate of the support of a probability
        ## density function. A dynamic process associated with the variance function
        ## can be built and applied to cluster labeling of the data points. The
        ## variance function estimates the support region by sigma^2 <= theta, where
        ## theta = max(sigma^2(x))
        ##
        ## Input:
        ##  input [dim x num_data] Input data.
        ##  hyperparams [(num_data + 2) x 1] 
        ##
        ## Output:
        ##  model [struct] Center of the ball in the kernel feature space:
        ##   .input [dim x num_data]
        ##   .X [num_data x dim]
        ##   .hyperparams [(num_data + 2) x 1]  
        ##   .inside_ind [1 x num_data] : all input points are required to be used for computing
        ##   support function value.
        ##   .inv_C [num_data x num_data] : inverse of a covariance matrix C for the training inputs
        ##   .r : support function level with which covers estimated support region 

        print("---------------------------------")
        print('Step 1 : Training Support Function by GP...')
        
        self.inside_ind = range(self.normalized_input.shape[1])
        self.get_inv_C()
        tmp = var_gpr(self.normalized_input, self.normalized_input, self.inv_C, self.gp_params) #######self
        self.R = max(tmp)

        print('Training Completed !')
        
        
    def get_inv_C(self):
        ## Compute inverse of a covariance matrix C

        ninput=self.normalized_input
        params=self.gp_params
        [D,n] = ninput.shape ## dimension of input space and number of training cases
        C = np.zeros([n,n])

        for d in range(D):    
            C = C + params[0][d]*(np.tile(np.reshape(ninput[d,:],[-1,1]),[1,n])-np.tile(ninput[d,:],[n,1]))**2
        C = params[1]*np.exp(-0.5*C) + params[2]*np.eye(n)
                    
        self.inv_C = inv(np.matrix(C))
        
    ####Bumho Son
    def svdd_normalize(self):
        Xin = self.input
        [dim, n] = Xin.shape
    
        mean_by_col = np.mean(Xin, axis=1).reshape(dim, 1)
        stds_by_col = np.std(Xin, axis=1).reshape(dim, 1)

        means = np.tile(mean_by_col, (1,n))
        stds = np.tile(stds_by_col, (1,n))

        X_normal = (Xin - means)/stds

        self.normalized_input = X_normal
        print("Normalized input : ", self.normalized_input)
    
    def svdd(self):
        print("Step 1: Training Support Function by SVDD ...")
        start_time = time.time()

        self.model = {}
        self.model['support'] = 'SVDD'
        options = self.svdd_params
        if options.get('ker') == None:
            options['ker'] = 'rbf'
        if options.get('arg') == None:
            options['arg'] = 1
        if options.get('solver') == None:
            options['solver'] = 'imdm'
        if options.get('C') == None:
            options['C'] = 1

        [dim, num_data] = self.normalized_input.shape

        # Set up QP Problem
        K = kernel(input1=self.normalized_input, ker=options['ker'], arg=options['arg'])
        f = -np.diag(K)
        H = 2 * K
        b = options['C']
        I = np.arange(num_data)
        [Alpha, fval] = qpssvm(H, f, b, I)  # 아직 qpssvm에서 stat 미구현

        inx = np.where(Alpha > pow(10, -5))[0]  # Alpha를 0이상으로 잡으면 잘못잡힘.
        self.model['support_type'] = 'SVDD'
        self.model['Alpha'] = Alpha[inx]
        self.model['sv_ind'] = np.where((Alpha > pow(10, -5)) & (Alpha < (options['C'] - pow(10, -7))))[0]
        print(self.model['sv_ind'])
        self.model['bsv_ind'] = np.where(Alpha >= (options['C'] - pow(10, -7)))[0]
        self.model['inside_ind'] = np.where(Alpha < (options['C'] - pow(10, -7)))[0]

        self.model['b'] = np.dot(np.dot(Alpha[inx].T, K[inx, :][:, inx]), Alpha[inx])

        # setup model
        self.model['sv'] = {}
        self.model['sv']['X'] = self.normalized_input[:, inx]

        self.model['sv']['inx'] = inx
        self.model['nsv'] = len(inx)
        self.model['options'] = options
        #    model['stat'] = stat
        self.model['fun'] = 'kradius'

        radius = kradius(self.normalized_input[:, self.model['sv_ind']], self.model)

        self.model['r'] = np.amax(radius)

        print("Training Completed!")
        end_time = time.time()
        print("Trading time for SVDD is : ", end_time - start_time, " sec")

    



class labeling:
    def __init__(self, supportmodel, labelingmethod, options=None):
        self.supportmodel = supportmodel
        self.labelingmethod = labelingmethod
        self.options = options

    def run(self):
        print("Step 2 : Labeling by the method "+self.labelingmethod+"...")
        if self.labelingmethod=='CG-SC':
            self.cgsc()
        elif self.labelingmethod=='S-MSC':
            self.smsc()       
        elif self.labelingmethod=='T-MSC':
            self.tmsc()     
        elif self.labelingmethod=='F-MSC':
            self.fmsc()    
        elif self.labelingmethod=='V-MSC':
            self.vmsc()
        else:
            print("ERROR: invalid labeling method; valid examples (CG-SC, S-MSC, T-MSC, F-MSC, V-MSC)")
        print("Labeling Completed!")


    def findAdjMatrix(self, input):
#%==========================================================================
#% FindAdjMatrix: Caculating adjacency matrix
#%
#% Input:
#%  X [dim x num_data] Input data.
#%  model [struct] obtained from svdd.m 
#%
#% Output:
#%  adjacent [num_data x num_data] 
#%     1 for connected, 0 for disconnected (violated), -1 (outliers, BSV)
#%
#% Description
#%	The Adjacency matrix between pairs of points whose images lie in
#%	or on the sphere in feature space. 
#%	(i.e. points that belongs to one of the clusters in the data space)
#%
#%	given a pair of data points that belong to different clusters,
#%	any path that connects them must exit from the sphere in feature
#%	space. Such a path contains a line segment of points y, such that:
#%	kdist2(y,model)>model.r.
#%	Checking the line segment is implemented by sampling a number of 
#%   points (10 points).
#%	
#%	BSVs are unclassfied by this procedure, since their feature space 
#%	images lie outside the enclosing sphere.( adjcent(bsv,others)=-1 )
#%
#%==========================================================================
#% January 13, 2009
#% Implemented by Daewon Lee
#%
#% * The source code is available under the GNU LESSER GENERAL PUBLIC
#% LICENSE, version 2.1. 
#%==========================================================================
#
#% samples are column vectors
        model = self.supportmodel
        N = input.shape[1]

        adjacent = np.zeros([N,N])
        R = model.model['r']+10**(-7)  #% Squared radius of the minimal enclosing ball
    
        for i in range(N): ##rows
            for j in range(N): ##columns
                ## if the j is adjacent to i - then all j adjacent's are also adjacent to i.
                if j == i:
                    adjacent[i,j] = 1
                elif j < i:
                    if (adjacent[i,j] == 1):
                        adjacent[i,:] = np.logical_or(adjacent[i,:],adjacent[j,:])
                        adjacent[:,i]=adjacent[i,:]
                else:
                    ## if adajecancy already found - no point in checking again
                    if (adjacent[i,j] != 1):
                    ## goes over 10 points in the interval between these 2 Sample points                   
                        adj_flag = 1 ## unless a point on the path exits the shpere - the points are adjacnet
                        for interval in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}:
                            z = input[:,i] + interval * (input[:,j] - input[:,i])
                            z = np.reshape(z,[-1,1])
                            ## calculates the sub-point distance from the sphere's center 
                            d = kradius(z, model.model)
                            if d > R:
                                adj_flag = 0
                                break
                        if adj_flag == 1:
                            adjacent[i,j] = 1
                            adjacent[j,i] = 1
                            
        self.adjacent_matrix = adjacent
        self.symmetric = (np.max(np.abs(adjacent-adjacent.T) == 0))


    
    def cgsc(self):
#% CGSVC Support Vector Clusteing using Complete-Graph Based Labeling Method
#%
#% Description:
#%  To determine whether a pair of xi and xj is in the same contour, 
#%  it can be used a complete-graph(CG) strategy that relies on the fact 
#%  that any path connecting two data points in different contours must 
#%  exit the contours in data space, which is equivalent that the image 
#%  of the path in feature space exits the minimum enclosing sphere.
#%  CG strategy makes adjacency matrix Aij between pairs of points 
#%  xi and xj as follows :
#%   A(ij) = 1 if for all y on the line segment connecting xi and xj
#%           0 otherwise
#%
#% * The source code is available under the GNU LESSER GENERAL PUBLIC
#% LICENSE, version 2.1. 
        model = self.supportmodel
        self.findAdjMatrix(model.normalized_input)
        self.cluster_label = cg.connected_components(self.adjacent_matrix)
        print(self.cluster_label[1])

    ### Junyoung Byun
    def findSEPs(self):
        model = self.supportmodel
        X = model.normalized_input

        [dim, N] = X.shape
        N_locals = []
        local_val = []

        if model.model['support_type'] == 'GP':
            for i in range(N):
                x0 = X[i]
                res = minimize(fun=my_R_GP, x0=x0)
                [temp, val] = [res.x, res.fun]
                N_locals.append(temp)
                local_val.append(val)

        if model.model['support_type'] == 'SVDD':
            for i in range(N):
                x0 = X.T[i]
                if len(x0) <= 2:
                    res = minimize(fun=my_R1, x0=x0, args=model.model, method='Nelder-Mead')
                    [temp, val] = [res.x, res.fun]
                else:
                    res = minimize(fun=my_R1, x0=x0, method='trust-ncg')
                    [temp, val] = [res.x, res.fun]
                N_locals.append(temp)
                local_val.append(val)
        N_locals = np.array(N_locals)
        local_val = np.array(local_val)
        local, I, match_local = np.unique(np.round(10 * N_locals), axis=0, return_index=True, return_inverse=True)

        newlocal = N_locals[I, :]
        newlocal_val = local_val[I]
        return [N_locals, newlocal, newlocal_val, match_local]

    def smsc(self):
        [rep_locals, locals, local_val, match_local] = self.findSEPs()
        # %% Step 2 : Labeling Data for Clustering

        self.findAdjMatrix(locals.T)

        # Finds the cluster assignment of each data point
        # clusters = findConnectedComponents(self.adjacent_matrix)
        csm = cg.connected_components(self.adjacent_matrix)
        local_cluster_assignments = csm[1]
        local_cluster_assignments = np.array(local_cluster_assignments)
        print(local_cluster_assignments[match_local])

    def findTPs(self):
        locals = self.locals
        model = self.supportmodel
        epsilon = self.options['epsilon']
        R = model.R + 10 ** (-7)
        print(self.locals)
        ts = {}
        ts['x'] = []
        ts['f'] = []
        ts['neighbor'] = []
        ts['purturb'] = []
        [N, attr] = locals.shape
        tmp_x = []

        if model.support == 'GP':
            for i in range(N):
                for j in range(i, N):
                    for k in range(10):
                        x0 = locals[i] + 0.1 * (k + 1) * (locals[j] - locals[i])
                        sep = fsolve(func=fsolve_R_GP, x0=x0, args=model, xtol=10 ** (-6))
                        tmp_x.append(sep)
            tmp_x = np.array(tmp_x)
            [dummy, I, J] = np.unique(np.round(10 * tmp_x), axis=0, return_index=True, return_inverse=True)
            tmp_x = tmp_x[I, :]
            for i in range(list(tmp_x.shape)[0]):
                sep = tmp_x[i]
                [f, g, H] = my_R_GP2(sep, model)
                [D, V] = la.eig(H)

                ind = []
                if np.sum(D < 0) == 1:
                    sep1 = sep + epsilon * V[np.where(D < 0)]
                    sep2 = sep - epsilon * V[np.where(D < 0)]

                    if attr == 2:
                        res1 = minimize(fun=my_R_GP1, x0=sep1, args=model, method='Nelder-Mead')
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R_GP1, x0=sep2, args=model, method='Nelder-Mead')
                        [temp2, val] = [res2.x, res2.fun]
                    else:
                        res1 = minimize(fun=my_R_GP1, x0=sep1, args=model, hess=True)
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R_GP1, x0=sep2, args=model, hess=True)
                        [temp2, val] = [res2.x, res2.fun]
                    [dummy, ind1] = [np.min(euclidean_distances(temp1.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp1.reshape(1, -1), locals))]
                    [dummy, ind2] = [np.min(euclidean_distances(temp2.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp2.reshape(1, -1), locals))]
                    if ind1 != ind2:
                        ts['x'].append(sep)
                        ts['f'].append(f)
                        ts['neighbor'].append([ind1, ind2])
                        ts['purturb'].append([sep1, sep2])
        if model.support == 'SVDD':
            for i in range(N):
                for j in range(i, N):
                    for k in range(10):
                        x0 = locals[i] + 0.1 * (k + 1) * (locals[j] - locals[i])
                        sep = fsolve(func=fsolve_R, x0=x0, args=model, maxfev=300, xtol=10 ** (-6))
                        tmp_x.append(sep)
            tmp_x = np.array(tmp_x)
            [dummy, I, J] = np.unique(np.round(10 * tmp_x), axis=0, return_index=True, return_inverse=True)
            tmp_x = tmp_x[I, :]

            for i in range(list(tmp_x.shape)[0]):
                sep = tmp_x[i]
                [f, g, H] = my_R2(sep, model)
                [D, V] = la.eig(H)
                ind = []
                if np.sum(D < 0) == 1:

                    sep1 = sep + epsilon * V[np.where(D < 0)]
                    sep2 = sep - epsilon * V[np.where(D < 0)]
                    if attr == 2:
                        res1 = minimize(fun=my_R1, x0=sep1, args=model, method='Nelder-Mead')
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R1, x0=sep2, args=model, method='Nelder-Mead')
                        [temp2, val] = [res2.x, res2.fun]
                    else:
                        res1 = minimize(fun=my_R1, x0=sep1, args=model, hess=True)
                        [temp1, val] = [res1.x, res1.fun]
                        res2 = minimize(fun=my_R1, x0=sep2, args=model, hess=True)
                        [temp2, val] = [res2.x, res2.fun]
                    [dummy, ind1] = [np.min(euclidean_distances(temp1.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp1.reshape(1, -1), locals))]
                    [dummy, ind2] = [np.min(euclidean_distances(temp2.reshape(1, -1), locals)),
                                     np.argmin(euclidean_distances(temp2.reshape(1, -1), locals))]
                    if ind1 != ind2:
                        ts['x'].append(sep)
                        ts['f'].append(f)
                        ts['neighbor'].append([ind1, ind2])
                        ts['purturb'].append([sep1, sep2])
        ts['x'] = np.array(ts['x'])
        ts['f'] = np.array(ts['f'])
        ts['neighbor'] = np.array(ts['neighbor'])
        ts['purturb'] = np.array(ts['purturb'])
        self.ts = ts

    def hierarchicalLabelTSVC(self):
        print("hierarchicalLableTSVC")
        nOfLocals = self.locals.shape[0]
        ts = self.ts
        nOfTS = len(ts['f'])
        K = self.options['K']

        local_clusters_assignments = []
        f_sort = np.sort(ts['f'], 0)  # small --> large
        print("f_sort:", f_sort)
        adjacent = np.zeros([nOfLocals, nOfLocals, nOfTS])
        a = []
        flag = 0
        for m in range(nOfTS):
            cur_f = f_sort[
                -m - 1]  # % cutting level:large --> small  (small number of clusters --> large number of clusters)
            # %cur_f=f_sort(i);         % cutting level: small --> large (large number of clusters --> small number of clusters)

            tmp = np.nonzero(ts['f'] < cur_f)[0]
            if len(tmp) > 0:  # % TSs inside the sphere
                for j in range(len(tmp)):
                    adjacent[ts['neighbor'][tmp[j], 0], ts['neighbor'][tmp[j], 1], m] = 1
                    adjacent[ts['neighbor'][tmp[j], 1], ts['neighbor'][tmp[j], 0], m] = 1
                    # %% To connect nodes which can be connected via directly connected edges.
                for i in range(nOfLocals):
                    for j in range(i):
                        if (adjacent[i, j, m] == 1):
                            adjacent[i, :, m] = np.logical_or(adjacent[i, :, m], adjacent[j, :, m])
                    adjacent[i, i] = 1

            a = [a, cur_f]
            my_ts = {}
            my_ts['x'] = ts['x'][tmp, :]
            my_ts['f'] = ts['f'][tmp, :]
            my_ts['purturb'] = ts['purturb'][tmp, :]
            my_ts['neighbor'] = ts['neighbor'][tmp, :]
            my_ts['cuttingLevel'] = cur_f
            ind = np.nonzero(ts['f'] == cur_f)[0]
            my_ts['levelx'] = ts['x'][ind[0], :]
            tmp_ts = {}  ####dictionary
            tmp_ts[m] = my_ts

            assignment = cg.connected_components(adjacent[:, :, m])[1]
            print("assignment:", assignment)
            print("N_clusters:", np.max(assignment) + 1)
            if np.max(assignment) == K - 1:
                print('We can find the number of K clusters');
                # % clstmodel update
                self.out_ts = tmp_ts[m]
                # % cluster assignment into entire data points
                self.local_ass = assignment
                self.cluster_labels = self.local_ass[self.match_local].T
                flag = 1
                break

            local_clusters_assignments = [local_clusters_assignments, assignment]

            # % cannot find k clusters
        if flag == 0:
            print(
                'Cannot find cluster assignments with K number of clusters, instead that we find cluster assignments the with the nearest number of clusters to K !');
            [dummy, ind] = np.min(euclidean_distances(np.max(local_clusters_assignments, 0).T, K), 0)  ####min/max

            # %ts=[];
            self.out_ts = tmp_ts[ind[0]]
            local_clusters_assignments = local_clusters_assignments[:, ind[0]]
            self.local_ass = local_clusters_assignments
            self.cluster_labels = self.local_ass[self.match_local]
            print(self.cluster_labels)
        
    def tmsc(self):
        fHierarchical = self.options['hierarchical']

        # % Find SEPs
        [rep_locals, locals, local_val, match_local] = self.findSEPs()
        nOfLocals = locals.shape[0]
        self.locals = locals  #####transpose
        self.match_local = match_local
        # % Find transition points and label the SEPs
        self.findTPs()

        # %% Cluster assignment of each data point

        # --- Automatic determination of cluster number based on the cluster boundary
        if not (fHierarchical):

            print('Automatic determination of cluster numbers based on the SVDD boundearies defined by R^2');
            adjacent = np.zeros([nOfLocals, nOfLocals])

            tmp = np.nonzero(self.ts['f'] < self.supportmodel.R)[0]  ########
            if np.nonzero(len(tmp)):  # % only check the connectivity of TSs inside the sphere
                for j in range(len(tmp)):
                    adjacent[self.ts['neighbor'][tmp[j], 0], self.ts['neighbor'][tmp[j], 1]] = 1
                    adjacent[self.ts['neighbor'][tmp[j], 1], self.ts['neighbor'][tmp[j], 0]] = 1
                    # %% To connect nodes which can be connected via directly connected edges.
                for i in range(nOfLocals):
                    for j in range(i):
                        if (adjacent[i, j] == 1):
                            adjacent[i, :] = np.logical_or(adjacent[i, :], adjacent[j, :])
                    adjacent[i, i] = 1
                self.local_clusters_assignments = np.array(cg.connected_components(adjacent)[1])
                print(self.local_clusters_assignments)
                # % model update
            self.ts['x'] = self.ts['x'][tmp, :]
            self.ts['f'] = self.ts['f'][tmp, :]
            self.ts['purturb'] = self.ts['purturb'][tmp, :]
            self.ts['neighbor'] = self.ts['neighbor'][tmp, :]
            self.ts['cuttingLevel'] = self.supportmodel.R

            # % cluster assignment into entire data points
            self.cluster_labels = self.local_clusters_assignments[match_local].T  ###transpose
            print(self.cluster_labels)
        else:
            self.hierarchicalLabelTSVC()




def my_R_GP(x, supportmodel, nargout=1):
##Usage of nargout
##if you need f only->nargout=1
##if you need f and gradient only->nargout=2
##if you need all f, gradient and Hessian->nargout=3

#%   Calculating function value and gradient of
#%   the trained kernel radius function
#%==========================================================================
#% Implemented by Kyu-Hwan Jung at April 26, 2010.
#% Modified by Sujee Lee at September 3, 2014.
#%
#% * The source code is available under the GNU LESSER GENERAL PUBLIC
#% LICENSE, version 2.1. 
#%==========================================================================
    model=supportmodel
    f= var_gpr(x, model.input, model.inv_C, model.gp_params) ##full(X)
#    %[n d]=size(model.SVs);
#    %q=model.Parameters(4);
#    %SV=full(model.SVs);
#    %beta=model.sv_coef;

    if nargout>1:
        #x=full(x)
        xt=x.T
        
        input = model.input
        hparam = model.gp_params
        tinput=input.T
        [D,N]=input.shape
        [D,nn]=x.shape # % nn=1
    
        inv_K = model.inv_C
        k = np.zeros([N, nn])  #% create and zero space
        for d in range(D):
            k = k + hparam[0][d]*(np.tile(np.reshape(tinput[:,d],[-1,1]),[1,nn])-np.tile(xt[:,d],[N,1]))**2 ## nn * N, last term, np.tie(x....), need to be changed
        k = hparam[1]*np.exp(-0.5*k)
                    
        gk = - np.tile(k,[1,D])*np.tile(hparam[0,:],[N,1])*(np.tile(xt,[N,1])-tinput)  #% N * D ##hparam[0:D] ##elementwise
        g = - 2 * np.matmul(np.matmul(gk.T , inv_K) , k) #D * nn
        #%     g= gradient(@(xx)var_gpr(xx,model.X,model.inv_C,model.hyperparams),x); %%
                                       
        if nargout>2:
            Hk = np.zeros(N, D**2) # % Hessian of k. (make N * D^2 matrix)
            for j in range(D):
                Hkj = gk*(-hparam[0][j]*np.tile(np.reshape(xt[j]-tinput[:,j],[-1,1]),1,D))  #% x(:,j)-input(:,j) : scalar - vector
                Hkj[:,j] = Hkj[:,j] - hparam[0][j]*k
                Hk[:,j*D:(j+1)*D-1] = Hkj
            H1 = np.matmul(np.matmul(Hk.T, inv_K), k)  #% D^2 * 1
            H1 = np.reshape(H1,[D,D])  #% D * D
            H2 = np.matmul(np.matmul(gk.T, inv_K),gk) #% D * D
            H = -2*(H1+H2)
            #%         H= hessian(@(xx)var_gpr(xx,model.X,model.inv_C,model.hyperparams),x); %%
    return f,g,H

def my_R1(x, model):
    f = kradius(x.reshape(x.shape[0], 1), model)
    return f

def my_R2(x, model):
    d = x.shape[0]
    n = model.svdd_model['nsv']
    f = kradius(x.reshape(x.shape[0], 1), model)

    q = 1 / (2 * model.svdd_model['options']['arg'] * model.svdd_model['options']['arg'])
    K = kernel(model.svdd_model['sv']['X'], model.svdd_model['options']['ker'], model.svdd_model['options']['arg'],
               input2=x.reshape(x.shape[0], 1))
    g = 4 * q * np.dot(model.svdd_model['Alpha'].reshape(model.svdd_model['Alpha'].shape[0], 1).T,
                       np.multiply(ml.repmat(K, 1, d),
                                   (ml.repmat(x, n, 1) - model.svdd_model['sv']['X'].T)))

    const = np.multiply(model.svdd_model['Alpha'], K)
    H = []

    for i in range(d):
        H.append(- 8 * q * q * np.sum(np.multiply(np.multiply(ml.repmat(const.T, d, 1), (
        ml.repmat(x[i], d, n) - ml.repmat(model.svdd_model['sv']['X'][i, :].T, d, 1))), (
                                                  ml.repmat(x.reshape(x.shape[0], 1), 1, n) -
                                                  model.svdd_model['sv']['X'])), axis=1).T)
    H = np.array(H).T
    H = H + 4 * q * np.eye(d) * np.dot(model.svdd_model['Alpha'].reshape(model.svdd_model['Alpha'].shape[0], 1).T, K)
    return f, g, H

def my_R_GP1(x, model):
    f = var_gpr(np.reshape(x, [-1, 1]), model.normalized_input, model.inv_C, model.gp_params)  ##full(X)
    return f

def my_R_GP2(x, model):
    # %
    # %   Calculating function value and gradient of
    # %   the trained kernel radius function
    # %==========================================================================
    # % Implemented by Kyu-Hwan Jung at April 26, 2010.
    # % Modified by Sujee Lee at September 3, 2014.
    # %
    # % * The source code is available under the GNU LESSER GENERAL PUBLIC
    # % LICENSE, version 2.1.
    # %==========================================================================
    x = x.reshape(-1, 1)

    input = model.normalized_input
    inv_K = model.inv_C
    hparam = model.gp_params

    f = var_gpr(x, input, inv_K, hparam)  ##############full

    input = input.T
    x = x.reshape(1, -1)
    hparam = np.append(hparam[0], np.array(hparam[1:]).reshape(-1, 1), axis=0)
    [N, D] = input.shape
    [nn, D] = x.shape  # % nn=1
    k = np.zeros((N, nn))

    for d in range(D):
        k += hparam[d][0] * np.power((ml.repmat(input[:, d].reshape(-1, 1), 1, nn) - ml.repmat(np.transpose(x[:, d]), N, 1)), 2)
    k = hparam[D] * np.exp(-0.5 * k)
    gk = -np.multiply(np.multiply(ml.repmat(k, 1, D), ml.repmat(np.transpose(hparam[0:D]), N, 1)), ml.repmat(x, N, 1) - input)
    g = -2 * np.dot(np.dot(np.transpose(gk), inv_K), k)

    Hk = np.zeros((N, D ** 2))  # % Hessian of k. (make N * D^2 matrix)
    for j in range(D):
        Hkj = np.multiply(gk , -hparam[j] * np.matlib.repmat((x[:, j] - input[:, j]), D, 1).T ) # % x(:,j)-input(:,j) : scalar - vector
        Hkj[:, j] = Hkj[:, j] - hparam[j] * k.reshape(1, -1)
        Hk[:, j * D:(j + 1) * D] = Hkj  #####################index
    H1 = np.matmul(np.matmul(Hk.T, inv_K), k)  # % D^2 * 1
    H1 = np.reshape(H1, [D, D])  # % D * D
    H2 = np.matmul(np.matmul(gk.T, inv_K), gk)  # % D * D
    H = -2 * (H1 + H2)
    # %         H= hessian(@(xx)var_gpr(xx,model.X,model.inv_C,model.hyperparams),x); %%
    return f, g, H

def fsolve_R(x, model):
    d = x.shape[0]
    n = model.svdd_model['nsv']

    q = 1 / (2 * model.svdd_model['options']['arg'] ** 2)
    K = kernel(model.svdd_model['sv']['X'], ker=model.svdd_model['options']['ker'],
               arg=model.svdd_model['options']['arg'], input2=x.reshape(x.shape[0], 1))

    F = 4 * q * np.dot(np.transpose(model.svdd_model['Alpha']),
                       np.multiply(ml.repmat(K, 1, d), ml.repmat(x, n, 1) - np.transpose(model.svdd_model['sv']['X'])))

    # if nargout >1:
    #    const = np.multiply(np.transpose(model['Alpha']),K)
    #    J = []
    #    for i in range(d):
    #    for i in range(d):
    #        J=[J-8*q**2*np.sum(np.multiply(np.transpose(ml.repmat(np.transpose(const),d,1))),ml.repmat(x[i],d,n)-np.multiply(np.transpose(ml.repmat(model['sv']['X'][i,:],d,1)),ml.repmat(np.transpose(x),1,n)-model['sv']['X']),1)]

    #   J = J + 4*q*np.multiply(np.multiply(np.eye(d), np.transpose(model['Alpha'])), K)

    return F[0]


def fsolve_R_GP(x, model):
    input = model.normalized_input.T
    hparam = model.gp_params

    [N, D] = input.shape
    x = x.reshape(-1, D)
    [nn, D] = x.shape

    inv_K = model.inv_C


    hparam = np.append(hparam[0], np.array(hparam[1:]).reshape(-1,1), axis=0)

    k = np.zeros((N, nn))
    for d in range(D):

        k += hparam[d][0] * np.power((ml.repmat(input[:, d].reshape(-1, 1), 1, nn)-ml.repmat(np.transpose(x[:, d]), N, 1)), 2)
    k = hparam[D] * np.exp(-0.5 * k)

    gk = -np.multiply(np.multiply(ml.repmat(k, 1, D), ml.repmat(np.transpose(hparam[0:D]), N, 1)),
                      ml.repmat(x, N, 1) - input)
    g = -2 * np.dot(np.dot(np.transpose(gk),inv_K), k)

    return g.ravel()

                                   


