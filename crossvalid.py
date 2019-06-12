"""
This code shows how to analyse predictions using 
learning curves to improve learning performance.
Auth : Zak Moktadir
"""

import matplotlib
import matplotlib.image as mpimg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
#from scipy.stats import multivariate_normal
import scipy.io as sio
from numpy import linalg as LA
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler



def LinearRegCostFunction(theta,X,y,reg_param=0.0, **kwargs):

    m,n = X.shape
    theta = theta.reshape([n, 1])
    J = np.sum(((X@theta)-y)**2)/(2*m) # non-regularized cost function

    J = J + reg_param*np.sum(theta**2)/(2*m) # regularized cost function

    return J

def gradientCostFunction(theta,X,y,reg_param=0.0):

    m,n = X.shape
    theta=theta.reshape([n,1])

# Gradient of the cost function

    grad0 = X.T @ (X @ theta - y) / m

    grad = grad0 + (reg_param * theta / m)

    grad[0]  = grad0[0]

    return grad.flatten()

def trainLinearReg(X,y,reg_param=0.0):

    m,n = X.shape
    #theta_ini=np.random.uniform(0, 1, [n, 1]).reshape([n,1])
    theta_ini=np.zeros([n,1])
    #theta_ini = np.zeros([n,1])
    res = minimize(LinearRegCostFunction, x0=theta_ini, args=(X, y, reg_param),method='TNC',
                   jac=gradientCostFunction,options={'maxiter':1000})

    return  res


# implement learning curves function

def learningcurves(X, y, Xval, yval, reg_param=0.0):
    m = X.shape[0]
    mval=Xval.shape[0]
    train_error = []
    val_error = []
    for i in np.arange(1,m):
        train_theta = trainLinearReg(np.c_[np.ones([i,1]),X[0:i,:]], y[0:i], reg_param)
        train_error.append(LinearRegCostFunction(train_theta.x,np.c_[np.ones([i,1]),X[0:i,:]], y[0:i], 0.0))
        val_error.append(LinearRegCostFunction(train_theta.x, np.c_[np.ones([mval,1]),Xval], yval, 0.0))

    return (train_error), (val_error)

def polyfeatures(X, p):
    Xpoly=X.copy()
    for i in range(1,p):
        Xpoly = np.c_[(Xpoly,X**(i+1))]
    return Xpoly

def normalise(X,object,options):
    scaler = object(options)
    scaler.fit_transform(X)

def validationCurve(X,Y,Xval,yval,reg_param=0.0):
    l=len(reg_param)
    m,n = X.shape
    train_error = []
    val_error = []
    mval = Xval.shape[0]


    for i in range(l):
        theta = trainLinearReg(X,y,reg_param[i])
        train_error.append(LinearRegCostFunction(theta.x,X,y,0.0))
        val_error.append(LinearRegCostFunction(theta.x,Xval, yval,0.0))

    return train_error, val_error



#load data
data=sio.loadmat('ex5data1.mat')
X=data['X']
Xval=data['Xval']
X=np.asanyarray(X, dtype=np.float32)
Xval=np.asanyarray(Xval,dtype=np.float32)
m,n = X.shape
y=data['y']
yval=data['yval']
y=np.asanyarray(y, dtype=np.float32)
yval=np.asanyarray(yval,dtype=np.float32)



#plt.plot(X,y,'x',X,np.c_[np.ones([m,1]),X]@res.x)
#plt.show()

train_error, val_error = learningcurves(X,y,Xval,yval,0.0)

plt.plot(range(m-1),train_error,range(m-1),val_error)
plt.show()
reg_param=4.0
p=8
# compute polynomial features and optimize to find the new parameter set
Xpoly=polyfeatures(X,p)
normalise(Xpoly,StandardScaler,options=False)
Xpoly = np.c_[np.ones([m,1]), Xpoly]  # add ones

res_poly = trainLinearReg(Xpoly, y, reg_param=reg_param)
best_theta = res_poly.x
print(best_theta)


# Check how good the fit is by plotting the new predicted values
xx = np.arange(min(X),max(X),0.05)
xx_poly=polyfeatures(xx,p)
normalise(xx_poly,StandardScaler,options=False)
xx_poly = np.concatenate((np.ones([len(xx_poly),1]),xx_poly),axis=1) # add 1
yy= xx_poly@best_theta

plt.plot(X,y,'x',xx,yy)

#learning curves:
Xvalpoly=polyfeatures(Xval,p)
normalise(Xvalpoly,StandardScaler,options=False)
Xvalpoly = np.c_[np.ones([len(Xvalpoly), 1]), Xvalpoly]  # add ones
train_error, val_error = learningcurves(Xpoly,y,Xvalpoly,yval,reg_param=reg_param)
plt.subplot(211)
plt.plot(range(m-1),train_error,range(m-1),val_error)


## Validation curves:
reg_param = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
train_error, val_error = validationCurve(Xpoly,y,Xvalpoly,yval,reg_param)

plt.subplot(212)
plt.plot(reg_param,train_error,reg_param,val_error)
plt.show()
