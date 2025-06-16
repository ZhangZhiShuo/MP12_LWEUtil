from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionLatticeSampler
from collections import deque
import random
import cupy as cp
import math
# from sage.all import matirx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag,svd
from scipy.stats import multivariate_normal

def generate_g(k):
    q=2**k
    print("q:",q)
    g_vec=[2**i for i in range(k)]
    # print("g:",g_vec)
    return g_vec
def generate_G(g_vec,n):
    g_vec_list=[]
    for i in range(n):
        g_vec_list.append(g_vec)
    G=block_diag(*g_vec_list)
    # print("G:",G)
    return cp.array(G)


def generate_Sk(k):
    Sk=np.zeros(shape=(k,k),dtype=int)
    for i in range(k):
        Sk[i,i]=2
        if i>=1:
            Sk[i,i-1]=-1
    print("Sk:",Sk)
    return Sk

def generate_S(S_k,n):
    Sk_list=[]
    for i in range(n):
        Sk_list.append(S_k)
    S=block_diag(*Sk_list)
    # print("S:\n",S)
    return S

def SampleR(size):
    choices=np.array([0,1,-1])
    proba=np.array([0.5,0.25,0.25])
    sampleR=np.random.choice(a=choices,p=proba,size=size)
    return sampleR
def TrapBaseGen(n,k,G,S):
    q=2**k
    # G=matrix(ZZ,G)
    # S=matrix(ZZ,S)
    omega=int(n)*int(k)
    # hat_m=int(n*math.log(2**k,10)+2*math.log(omega,10))
    hat_m=int(2*n*math.log10(2**k))
    # hat_m=int(n*k)
    m=hat_m+omega
    print("-------n------------------:\n",n)
    print("--------------hat_m-----------------:\n",hat_m)
    print("----------------m-------------------:\n",m)
    # D=DiscreteGaussianDistributionIntegerSampler(sigma=1,algorithm="uniform+online")
    A_hat=cp.random.randint(0,q,(n,hat_m),dtype=int)
    # print("A_hat:\n",A_hat)
    print("------------A_hat generated--------------\n")
    # R=cp.zeros((hat_m,omega),dtype=int)
    # for i in range(hat_m):
    #     for j in range(omega):
    #         # R[i,j]=D()
    #         R[i,j]=SampleR()
    R=cp.array(SampleR((hat_m,omega)))
    # print("---------R-----------:\n",R)
    print("--------------R generated--------------------\n")
    A_1=G-cp.dot(A_hat,R)
    # print("A_1:\n",A_1)
    A_1=A_1%q
    print("--------------A_1 generated--------------------\n")
    A=cp.hstack((A_hat,A_1))
    # A=block_matrix([[matrix(A_hat),matrix(A_1)]])
    print("-----------A generated-------------------\n")
    # print("A:\n",A)

    # G_pinv=G.pseudoinverse()
    # print("--------G_pinv col,row:",G_pinv.ncols(),G_pinv.nrows())
    # print("---------------------------\n",-matrix(ZZ,A_hat))
    # print()
    # W=G_pinv*(-matrix(ZZ,A_hat))
    # W=matrix(qset,W)
    # W=matrix(ZZ,W)
    W=solveW(A_hat=A_hat.get(),k=k)
    # print("----------------------------W:\n", W)
    I1=cp.identity(hat_m,dtype=int)
    Base11=cp.dot(cp.array(R),cp.array(W))+I1
    # Base11=matrix(cp.asnumpy(Base11))
    # print("------------Base11---------\n",Base11)
    Base12=cp.dot(cp.array(R),cp.array(S))
    # Base12=matrix(cp.asnumpy(Base12))
    # print("---------------Base12-------------\n", Base12)
    Base21=W
    Base22=S
    BaseA=np.block([[Base11.get(),Base12.get()],[Base21,Base22]])
    # BaseA=block_matrix([[Base11,Base12],[Base21,Base22]])
    # print("------------------BaseA------------------:\n",BaseA)
    # BaseA=cp.array(np.array(BaseA))
    # A=cp.array(np.array(A))
    # print("========verifybase=======:\n",cp.dot(A,BaseA)%q)
    return A,R,A_hat,BaseA

def solveW(A_hat,k):
    W_list=[]
    for i in range (A_hat.shape[0]):
        W_list.append(np.zeros(shape=(k,A_hat.shape[1]),dtype=int))
    for i in range(A_hat.shape[0]):
        for j in range(A_hat.shape[1]):
            val_solve=A_hat[i,j]
            binary_str=bin(val_solve)[2:]
            binary_list=[int(digit) for digit in binary_str]
            length=k-len(binary_list)
            for _ in range(length):
                binary_list=[0]+binary_list
            # print("==========binary_list==========:\n",binary_list)
            for k_i in range(k):
                W_list[i][k_i,j]=binary_list[k-1-k_i]
    MatrixW_list=[]
    for W_i in W_list:
        MatrixW_list.append([W_i])
    matrixW=np.block(MatrixW_list)
    print("----------matrixW-------:\n",-matrixW)
    return -matrixW


    return x_vec
def TrapGen(n,k,G):
    q=2**k
    # G=matrix(ZZ,G)
    # S=matrix(ZZ,S)
    omega=int(n)*int(k)
    # hat_m=int(n*math.log(2**k,10)+2*math.log(omega,10))
    hat_m=int(2*n*math.log10(2**k))
    # hat_m=int(n*k)
    m=hat_m+omega
    print("-------n------------------:\n",n)
    print("--------------hat_m-----------------:\n",hat_m)
    print("----------------m-------------------:\n",m)
    # D=DiscreteGaussianDistributionIntegerSampler(sigma=1,algorithm="uniform+online")
    A_hat=cp.random.randint(0,q,(n,hat_m),dtype=int)
    # print("A_hat:\n",A_hat)
    print("------------A_hat generated--------------\n")
    # R=cp.zeros((hat_m,omega),dtype=int)
    # for i in range(hat_m):
    #     for j in range(omega):
    #         # R[i,j]=D()
    #         R[i,j]=SampleR()
    R=cp.array(SampleR((hat_m,omega)))
    # print("---------R-----------:\n",R)
    print("--------------R generated--------------------\n")
    A_1=G-cp.dot(A_hat,R)
    # print("A_1:\n",A_1)
    A_1=A_1%q
    print("--------------A_1 generated--------------------\n")
    A=cp.hstack((A_hat,A_1))
    # A=block_matrix([[matrix(A_hat),matrix(A_1)]])
    print("-----------A generated-------------------\n")
    # print("A:\n",A)

    # G_pinv=G.pseudoinverse()
    # print("--------G_pinv col,row:",G_pinv.ncols(),G_pinv.nrows())
    # print("---------------------------\n",-matrix(ZZ,A_hat))
    # print()
    # W=G_pinv*(-matrix(ZZ,A_hat))
    # W=matrix(qset,W)
    # W=matrix(ZZ,W)
    # W=solveW(A_hat=matrix(A_hat),k=k)
    # print("----------------------------W:\n", W)
    # I=cp.identity(hat_m,dtype=int)
    # Base11=cp.dot(cp.array(R),cp.array(np.array(W)))+I
    # Base11=matrix(cp.asnumpy(Base11))
    # print("------------Base11---------\n",Base11)
    # Base12=cp.dot(cp.array(R),cp.array(np.array(S)))
    # Base12=matrix(cp.asnumpy(Base12))
    # print("---------------Base12-------------\n", Base12)
    # Base21=W
    # Base22=S
    # BaseA=block_matrix([[Base11,Base12],[Base21,Base22]])
    # print("------------------BaseA------------------:\n",BaseA)
    # BaseA=cp.array(np.array(BaseA))
    # A=cp.array(np.array(A))
    # print("========verifybase=======:\n",cp.dot(A,BaseA)%q)
    return A,R,A_hat
def TrapGenObs(n,k,G):
    q=2**k
    # G=matrix(ZZ,G)
    # S=matrix(ZZ,S)
    omega=int(n)*int(k)
    # hat_m=int(n*math.log(2**k,10)+2*math.log(omega,10))
    hat_m=int(2*n*math.log10(2**k))
    # hat_m=int(n*k)
    m=hat_m+omega
    print("-------n------------------:\n",n)
    print("--------------hat_m-----------------:\n",hat_m)
    print("----------------m-------------------:\n",m)
    # D=DiscreteGaussianDistributionIntegerSampler(sigma=1,algorithm="uniform+online")
    A_hat=cp.random.randint(0,q,(n,hat_m),dtype=int)
    # print("A_hat:\n",A_hat)
    print("------------A_hat generated--------------\n")
    # R=cp.zeros((hat_m,omega),dtype=int)
    # for i in range(hat_m):
    #     for j in range(omega):
    #         # R[i,j]=D()
    #         R[i,j]=SampleR()
    R=cp.array(SampleR((hat_m,omega)))
    # print("---------R-----------:\n",R)
    print("--------------R generated--------------------\n")
    A_1=G-cp.dot(A_hat,R)
    # print("A_1:\n",A_1)
    A_1=A_1%q
    print("--------------A_1 generated--------------------\n")
    A=cp.hstack((A_hat,A_1))
    # A=block_matrix([[matrix(A_hat),matrix(A_1)]])
    print("-----------A generated-------------------\n")
    # print("A:\n",A)

    # G_pinv=G.pseudoinverse()
    # print("--------G_pinv col,row:",G_pinv.ncols(),G_pinv.nrows())
    # print("---------------------------\n",-matrix(ZZ,A_hat))
    # print()
    # W=G_pinv*(-matrix(ZZ,A_hat))
    # W=matrix(qset,W)
    # W=matrix(ZZ,W)
    # W=solveW(A_hat=matrix(A_hat),k=k)
    # print("----------------------------W:\n", W)
    # I=cp.identity(hat_m,dtype=int)
    # Base11=cp.dot(cp.array(R),cp.array(np.array(W)))+I
    # Base11=matrix(cp.asnumpy(Base11))
    # print("------------Base11---------\n",Base11)
    # Base12=cp.dot(cp.array(R),cp.array(np.array(S)))
    # Base12=matrix(cp.asnumpy(Base12))
    # print("---------------Base12-------------\n", Base12)
    # Base21=W
    # Base22=S
    # BaseA=block_matrix([[Base11,Base12],[Base21,Base22]])
    # print("------------------BaseA------------------:\n",BaseA)
    # BaseA=cp.array(np.array(BaseA))
    # A=cp.array(np.array(A))
    # print("========verifybase=======:\n",cp.dot(A,BaseA)%q)
    I2=cp.identity(R.shape[1],dtype=int)
    bias=cp.vstack((R,I2))
    obs_list=[]
    # print("------------obs---------------")
    for k in range(3*bias.shape[0]):
        # print("------------obs---------------")
        index=np.random.randint(low=0,high=bias.shape[0],size=2)
        obs_list.append(index)
        
        A[:,[index[0],index[1]]]=A[:,[index[1],index[0]]]
        bias[[index[0],index[1]]]=bias[[index[1],index[0]]]
    print("-------bias--------------\n",bias)
    print("-------------test_obs------------------------\n",np.array_equal((cp.dot(A,bias)%q).get(),G.get()))
    return A,A_hat,R,bias,obs_list