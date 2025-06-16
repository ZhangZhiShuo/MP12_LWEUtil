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
def custom_round_to_nearest_int(num):
    de_part=num%1
    print("---------de_part------------\n",de_part)
    if de_part>=0.7:
        return math.ceil(num)
    else:
        return math.floor(num)
def is_even(number):
    return number % 2 == 0

def is_odd(number):
    return number % 2 != 0
def SampleInt():
    choices=np.array([0,-1,1])
    proba=np.array([0.5,0.25,0.25])
    sampleint=np.random.choice(a=choices,p=proba)
    return int(sampleint)
def table1establish(num,mode=0):
    SampleInt=DiscreteGaussianDistributionIntegerSampler(sigma=1,tau=3,algorithm="uniform+table",precision="mp")
    result_queue=deque()
    i=1
    if mode==0:
        while i<=num:
            sampleInt=1
            while is_odd(sampleInt):
                sampleInt=SampleInt()
            result_queue.append(sampleInt)
            i+=1
    else:
        while i<=num:
            sampleInt=0
            while is_even(sampleInt):
                sampleInt=SampleInt()
            result_queue.append(sampleInt)
            i+=1
    return result_queue
def print_queue(myqueue):
    for item in myqueue:
        print(item, end=" ")
def SampleX(k,u):
    X=[]
    table0=table1establish(num=k+6,mode=0)
    table1=table1establish(num=k+6, mode=1)
    u=int(u)
    for i in range(k):
        if is_even(u):
            xi=table0.pop()
        else:
            xi=table1.pop()
        X.append(xi)
        u=(u-xi)/2 
    print("X:",X)   
    return X

# def SampleX(k,u):
#     X=[]
#     u=int(u)
#     for i in range(k):
#         if is_even(u):
#             xi=0
#         else:
#             xi=np.random.choice(a=[-1,1],p=[0.5,0.5])
#         X.append(xi)
#         u=(u-xi)/2 
#     # print("X:",X)   
#     return X
def SampleXVec(u_vec,type="np",k=24):
    X_list=[]
    for u in u_vec:
        # print(u)
        # table0=table1establish(num=k+6,mode=0)
        # table1=table1establish(num=k+6, mode=1)
        x_vec=SampleX(k=k,u=u)
        X_list.append(x_vec)
    if type=="np":
        z_vec=np.hstack(tuple(X_list))
        return z_vec
    else:
        z_vec=cp.hstack(tuple(X_list))
        return z_vec
# def generate_u(k):
#     q=2**k
#     Uset=IntegerModRing(q)
#     u=Uset.random_element()
#     print("u:",u)
#     return u
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
def TrapGenObs(n,k,G,S):
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
def GPVSamplePre(A,BaseA,sigma,u):
    t=np.solve(A,u)
    n=A.shape[0]
    norm=np.linalg.norm(BaseA)
    sigma=norm*(math.sqrt(math.log(n))**1.36)
    D = DiscreteGaussianDistributionLatticeSampler(BaseA, sigma=sigma,c=-t)
    x_0=np.array(D())
    x_vec=x_0+t
    # print("--------------gpv___________x_vec___-------------------\n",x_vec)
    draw_proba(prob_vec=x_vec,name="gpvvector")
def discrete_gaussian_sample(cov,th,dimension):
    # sample=np.random.multivariate_normal(mean=np.zeros(shape=cov.shape[0]),cov=cov)
    sampleObj=multivariate_normal(mean=np.zeros(shape=cov.shape[0]),cov=cov)
    sample=sampleObj.rvs(size=cov.shape[0])
    print("-----------------门限-----------------\n",th)
    # D=DiscreteGaussianDistributionIntegerSampler(sigma=th,tau=1,algorithm="uniform+online")
    # # sample=cp.clip(a=sample,a_min=-th,a_max=th)
    for i in range(sample.shape[0]):
        if sample[i]<-th:
            sample[i]=int(sample[i]%(-th))
        elif sample[i]>th:
            sample[i]=int(sample[i]%(th))
        else:
            sample[i]=int(sample[i])
    # truncated_sample = cp.where((sample < -5*th) | (sample > 5*th), 0, sample)
    # truncated_sample=np.fix(sample).astype(int)
    # truncated_sample=sample.astype(int)
    # print("-----------samplepvec-----------------\n",truncated_sample.dtype,truncated_sample,np.max(truncated_sample))
    return sample
def discrete_gaussian_sample_obs(k,n,sigma,R,bias,obs_list):
    q=2**k
    ran_vec=np.random.randint(low=0,high=q,size=n,dtype=int)
    X_list=[]
    for u in ran_vec:
        # print(u)
        # table0=table1establish(num=k+6,mode=0)
        # table1=table1establish(num=k+6, mode=1)
        x_vec=SampleX(k=k,u=u)
        X_list.append(x_vec)
    z_vec=np.hstack(tuple(X_list))
    I1=np.identity(R.shape[1],dtype=int)
    bias_base=np.vstack((R.get(),I1))
    bias_vec=np.dot(bias_base,z_vec)
    bias_vec_R=bias_vec[0:R.shape[0]]
    print("-------------bias_vec_R------------\n",bias_vec_R)
    draw_proba(cp.array(bias_vec_R),name="dis_bias_vec_R")
    mean=np.mean(bias_vec_R)
    var=np.var(bias_vec_R)
    print("---{0}-----mean----var------{1}".format(mean,var))
    samvar=np.random.randint(low=1.5*var,high=3*var,dtype=int)
    # D=DiscreteGaussianDistributionIntegerSampler(sigma=math.sqrt(samvar),tau=3,algorithm="uniform+table",c=0)
    D=DiscreteGaussianDistributionIntegerSampler(sigma=math.sqrt(samvar),tau=3,algorithm="uniform+table",c=0,precision="mp")
    sample0=[]
    for _ in range(bias.shape[1]):
        sample0.append(D())
    sample0=np.array(sample0)
    draw_proba(cp.array(sample0),name="dis_sampleI")

    sample1=[]
    D2=DiscreteGaussianDistributionIntegerSampler(sigma=math.sqrt(samvar-var),tau=3,algorithm="uniform+table",c=-mean)
    for _ in range(R.shape[0]):
        # means=np.random.randint(-mean-3,-mean+3,dtype=int)
        
        sample1.append(D2())
    sample1=np.array(sample1)
    draw_proba(cp.array(sample1),name="dis_sampleR")
    sample=np.hstack((sample1,sample0))
    print("-----------------sample[0]---------------",sample.shape[0])
    for index in obs_list:
        # print(index)
        sample[index[0]],sample[index[1]]=sample[index[1]],sample[index[0]]
    # draw_proba(cp.array(sample1),name="dis_sampleR")
    # print("----sample0---\n",sample0)
    # print("-----sample1-----\n",list(sample1))
    # print("-----------------门限-----------------\n",th)
    # sample=sample0-cp.array(sample1)
    # D=DiscreteGaussianDistributionIntegerSampler(sigma=th,tau=1,algorithm="uniform+online")
    # # sample=cp.clip(a=sample,a_min=-th,a_max=th)
    # for i in range(sample.shape[0]):
    #     if sample[i]<-th:
    #         sample[i]=int(sample[i]%(-th))
    #     elif sample[i]>th:
    #         sample[i]=int(sample[i]%(th))
    #     else:
    #         sample[i]=int(sample[i])
    # truncated_sample = cp.where((sample < -5*th) | (sample > 5*th), 0, sample)
    truncated_sample=np.fix(sample).astype(int)

    print("-----------samplepvec-----------------\n",truncated_sample.dtype,truncated_sample,np.max(truncated_sample))
    return truncated_sample,samvar
def SamplePreObsPro(A_hat,A,u_vec,k,Bias,R,obs_list):
    m=A.shape[1]
    n=A.shape[0]
    q=2**k
    m=A.shape[1]
    bias=Bias
    # cov_p_1=0.5*np.dot(bias,bias.T)
    # print("============cov_p_1=============:\n",cov_p_1)
    # print("======================================q==================:\n",q)
    # Sigma=(s_1_R**2+1)*(s_1_G+2)
    # s=svd(a=math.sqrt(0.5)*bias,compute_uv=False)
    # print("--------------------------sqrt(Sigma)------------------\n",s[0])
    # SR=svd(a=R.get(),compute_uv=False)
    # print(type(s))
    # s_1_R=SR[0]
    # s_1_G=4
    # r=4.5
    # print("------------------s_1(R)--------------:\n",SR)
    # Sigma=(s_1_R+6)**2
    # Sigma0=s_1_R*2
    # Sigma=math.sqrt(n*n*k)*(math.log2(n)**2)
    # Sigma=Sigma0**2
    # Sigma=2*cp.max(cp.diag(cov_p_1))+6
    # print("--------------Sigma0,Sigma-----------------\n",Sigma0,Sigma)
    # print("-----------Sigma--------------------:\n",Sigma)


    # cov_p_yes=Sigma*cp.eye(N=cov_p_1.shape[0],M=cov_p_1.shape[1],dtype=int)-cov_p_1



    # test_cov_p=cov_p-cov_p_1
    # print("===================test_p_cov================\n",test_cov_p)
    # test_eigvals=np.linalg.eigvals(test_cov_p.get())
    # is_semidefinite=all(e>=0 for e in test_eigvals)
    # print("------------test_cov_p_ei-----------:\n",is_semidefinite)
    # is_semidefinite=all(e>=0 for e in test_eigvals)



    # print("--------cov_p-------------:\n",cov_p_yes)


    # eigvals, eigvectors=np.linalg.eig(cov_p.get())
    # print("------------cov_p_ei-----------:\n",eigvals)
    # is_semidefinite=all(e>=0 for e in eigvals)
    # print("----------------------cov_p_ei---------------------:\n",is_semidefinite)
    # sqrt_eigvals=np.sqrt(eigvals)
    # sqrt_cov_p=np.dot(eigvectors,np.dot(np.diag(sqrt_eigvals),eigvectors.T))
    # print("=================sqrt_cov_p=============================:\n",sqrt_cov_p)
    # print((r**2)*cov_p)
    # p_vec=discrete_gaussian_sample(mu= cp.zeros(shape=m),cov=cov_p,dimension=m,th=R.shape[1]/2.5)
    n=A.shape[0]
    # norm=np.max(np.linalg.norm(B,ord=2,axis=0))
    # sigma=norm*(math.sqrt(math.log(n))**1.36)
    # print("-----------sigmaInInIn-----------\n",sigma)
    # cov=s_1_R*np.identity(n=R.shape[0])
    p_vec,Sigma=discrete_gaussian_sample_obs_pro(k=k,n=n,R=R,bias=bias,obs_list=obs_list)

    # sample0=cp.array(sample0)
    # sample10=cp.array(sample10)
    # print("------p_vec-----------\n")
    # p_vec=vector(ZZ,p_vec)
    # A=matrix(ZZ,A)
    # print("---??",A*p_vec)
    u_vec=cp.array(u_vec,dtype=int)
    p_vec=cp.array(p_vec,dtype=int)
    vec_v_0=cp.dot(A,p_vec)%q
    # print("-----------vec_v_0----------------\n",vec_v_0)
    vec_v=(u_vec-vec_v_0)%q
    # print("---------vec_v----------\n",vec_v)
    # print("?????",R.shape[0],R.shape[1])

    z_vec=SampleXVec(vec_v,type="cp")
    # z_vec=vector(ZZ,z_vec)
    # print("=================z_vec=================:\n",z_vec)
    x_vec_0=cp.dot(bias,z_vec)
    draw_proba(x_vec_0[0:bias.shape[0]-bias.shape[1]],name="x_vec0R")
    draw_proba(x_vec_0[bias.shape[0]-bias.shape[1]:x_vec_0.shape[0]],name="x_vec0I")
    
    # print("-----------x_vec_0-----------\n",x_vec_0)
    x_vec=cp.array(p_vec+x_vec_0,dtype=int)
    # bias_test_vec=cp.array(x_vec_0[0:R.shape[0]]+sample10)
    # draw_proba(bias_test_vec,name="bias_test_vec")
    # print("-----------x_vec---------------\n",list(x_vec))
    print("------{0}-------x-vec-length:------{1}---\n".format(cp.linalg.norm(x_vec),math.sqrt(Sigma)*math.sqrt(m)))
    print("===============\\beta=============",A.shape[1]**1.5)
    verify=cp.dot(A,x_vec)%q
    print("==================Verify===========\n")
    print(np.array_equal(verify.get(),u_vec.get()))
    # print(u_vec)
    e_vec=Derror(sigma=math.sqrt(Sigma),c=0,n=x_vec.shape[0],tau=6)
    draw_proba(x_vec[0:bias.shape[1]-bias.shape[0]],name="sampleR")
    draw_proba(x_vec[bias.shape[1]-bias.shape[0]:x_vec.shape[0]],name="sampleI")
    draw_proba(x_vec,name='sample')
    draw_proba(cp.array(e_vec),name='error')
    return x_vec,Sigma
def discrete_gaussian_sample_obs_pro(k,n,R,bias,obs_list):
    q=2**k
    # ran_vec=np.random.randint(low=0,high=q,size=n,dtype=int)
    # for i in range(1,n-1):
    #     R[:,[i,i*k]]=R[:,[i*k,i]]
    # R1=R[:,0:n]
    # R2=R[:,n+1:n*k]
    # cov1=0.856*cp.dot(R1,R1.T)
    # cov2=cp.dot(R2,R2.T)
    # cov=cov1+cov2
    ran_vec=np.random.randint(low=0,high=q,size=n,dtype=int)
    test_vec=SampleXVec(u_vec=ran_vec,type="np")
    varDG=np.var(test_vec)
    meanDG=np.mean(test_vec)
    print("--------meanDG------varDG---------------------\n",meanDG,varDG)
    bias=cp.array(bias)
    s=svd((math.sqrt(varDG)*R).get(),full_matrices=False,compute_uv=False)
    cov=varDG*cp.dot(R,R.T)
    # eigvals, eigvectors=np.linalg.eig(s.get())
    Sigma=(np.max(s)+0.01)**2
    print("--------------Sigma----------------\n",Sigma)
    covp1=Sigma*np.identity(n=cov.shape[0])-cov.get()
    print("---------------covp1-----------------\n",covp1)
    meanVec=cp.dot(R,meanDG*cp.ones(shape=R.shape[1]))
    print("-----------------meanVec----------------\n",meanVec)
    sample0=np.random.multivariate_normal(mean=-meanVec.get(),cov=covp1)
    sample0int=[]
    for i in range(cov.shape[0]):
        sample0int.append(custom_round_to_nearest_int(sample0[i]))
    sample0=np.array(sample0int)
    draw_proba(cp.array(sample0),name="dis_sampleR")
    sample1=[]
    D=DiscreteGaussianDistributionIntegerSampler(sigma=math.sqrt(Sigma-varDG),tau=6,algorithm="uniform+table",c=-meanDG,precision="mp")
    for i in range(n*k):
        sample1.append(D())
    draw_proba(cp.array(sample1),name="dis_sampleI")

    sample=np.hstack((sample0,np.array(sample1)))
    for index in obs_list:
        # print(index)
        sample[index[0]],sample[index[1]]=sample[index[1]],sample[index[0]]
    # truncated_sample=np.fix(sample).astype(int)

    print("-----------samplepvec-----------------\n",sample.dtype,sample,np.max(sample))
    return sample,Sigma
def discrete_gaussian_sample3(sigma,cov,th,dimension):
    D=DiscreteGaussianDistributionIntegerSampler(sigma=sigma,algorithm="uniform+table",c=0)
    sample0=[]
    for _ in range(dimension):
        sample0.append(D())
    sample0=cp.array(sample0)
    # print("----sample0---\n",sample0)
    print("----sample0---\n",sample0)
    sample1=np.random.multivariate_normal(mean=np.zeros(shape=cov.shape[0]),cov=cov.get())
    # print("-----sample1-----\n",list(sample1))
    print("-----------------门限-----------------\n",th)
    sample=sample0-cp.array(sample1)
    # D=DiscreteGaussianDistributionIntegerSampler(sigma=th,tau=1,algorithm="uniform+online")
    # # sample=cp.clip(a=sample,a_min=-th,a_max=th)
    # for i in range(sample.shape[0]):
    #     if sample[i]<-th:
    #         sample[i]=int(sample[i]%(-th))
    #     elif sample[i]>th:
    #         sample[i]=int(sample[i]%(th))
    #     else:
    #         sample[i]=int(sample[i])
    # truncated_sample = cp.where((sample < -5*th) | (sample > 5*th), 0, sample)
    truncated_sample=cp.fix(sample).astype(int)

    print("-----------samplepvec-----------------\n",truncated_sample.dtype,truncated_sample,cp.max(truncated_sample))
    return truncated_sample

def Sample101(size):
    choices=np.array([0,-1,1])
    proba=np.array([0.5,0.25,0.25])
    sample101=np.random.choice(a=choices,p=proba,size=size)
    return sample101

def discrete_gaussian_sample2(sigma,cov,th,dimension):
    D=DiscreteGaussianDistributionIntegerSampler(sigma=sigma,algorithm="uniform+table",c=0,)
    sample0=[]
    for _ in range(dimension):
        sample0.append(D())
    sample0=np.array(sample0)
    # print("----sample0---\n",sample0)
    # print("----sample0---\n",sample0)
    draw_proba(cp.array(sample0),name="dis_sample0")
    sample10=np.random.multivariate_normal(mean=np.zeros(shape=cov.shape[0]),cov=cov)
    draw_proba(cp.array(sample10),name="dis_sample10")
    size=sample0.shape[0]-sample10.shape[0]
    sample11=Sample101(size=size)
    sample1=np.hstack((sample10,sample11))
    # print("-----sample1-----\n",list(sample1))
    print("-----------------门限-----------------\n",th)
    sample=sample0-sample1
    # D=DiscreteGaussianDistributionIntegerSampler(sigma=th,tau=1,algorithm="uniform+online")
    # # sample=cp.clip(a=sample,a_min=-th,a_max=th)
    # for i in range(sample.shape[0]):
    #     if sample[i]<-th:
    #         sample[i]=int(sample[i]%(-th))
    #     elif sample[i]>th:
    #         sample[i]=int(sample[i]%(th))
    #     else:
    #         sample[i]=int(sample[i])
    # truncated_sample = cp.where((sample < -5*th) | (sample > 5*th), 0, sample)
    truncated_sample=np.fix(sample).astype(int)

    print("-----------samplepvec-----------------\n",truncated_sample.dtype,truncated_sample,np.max(truncated_sample))
    return truncated_sample,sample0, sample10
def SamplePre(R,A_hat,A,u_vec,k,B=None):
    m=A.shape[1]
    n=A.shape[0]
    q=2**k
    # qset=Zmod(q)
    I=cp.identity(R.shape[1],dtype=int)
    U,s,Vt=np.linalg.svd(R.get())
    print(type(s))
    s_1_R=s[0]
    s_1_G=4
    r=4.5
    print("------------------s_1(R)--------------:\n",s)
    m=A.shape[1]
    cov_p_0=0.5*cp.dot(R,R.T)
    bias=cp.vstack((R,I))
    cov_p_1=0.5*cp.dot(bias,bias.T)
    print("============cov_p_1=============:\n",cov_p_1)
    # print("======================================q==================:\n",q)
    # Sigma=(s_1_R**2+1)*(s_1_G+2)
    U,s,Vt=np.linalg.svd(math.sqrt(0.5)*bias.get())
    print("--------------------------sqrt(Sigma)------------------\n",s[0])
    Sigma=(s[0]+0.06)**2
    # Sigma0=s_1_R*2
    # Sigma=math.sqrt(n*n*k)*(math.log2(n)**2)
    # Sigma=Sigma0**2
    # Sigma=2*cp.max(cp.diag(cov_p_1))+6
    # print("--------------Sigma0,Sigma-----------------\n",Sigma0,Sigma)
    print("-----------Sigma--------------------:\n",Sigma)
    cov_p_yes=Sigma*cp.eye(N=cov_p_1.shape[0],M=cov_p_1.shape[1],dtype=int)-cov_p_1
    # test_cov_p=cov_p-cov_p_1
    # print("===================test_p_cov================\n",test_cov_p)
    # test_eigvals=np.linalg.eigvals(test_cov_p.get())
    # is_semidefinite=all(e>=0 for e in test_eigvals)
    # print("------------test_cov_p_ei-----------:\n",is_semidefinite)
    # is_semidefinite=all(e>=0 for e in test_eigvals)
    print("--------cov_p-------------:\n",cov_p_yes)
    eigvals, eigvectors=np.linalg.eig(cov_p_yes.get())
    # print("------------cov_p_ei-----------:\n",eigvals)
    is_semidefinite=all(e>=0 for e in eigvals)
    print("----------------------cov_p_test---------------------:\n",is_semidefinite)
    # sqrt_eigvals=np.sqrt(eigvals)
    # sqrt_cov_p=np.dot(eigvectors,np.dot(np.diag(sqrt_eigvals),eigvectors.T))
    # print("=================sqrt_cov_p=============================:\n",sqrt_cov_p)
    # print((r**2)*cov_p)
    # p_vec=discrete_gaussian_sample(mu= cp.zeros(shape=m),cov=cov_p,dimension=m,th=R.shape[1]/2.5)
    n=A.shape[0]
    # norm=np.max(np.linalg.norm(B,ord=2,axis=0))
    # sigma=norm*(math.sqrt(math.log(n))**1.36)
    # print("-----------sigmaInInIn-----------\n",sigma)
    print(type(s_1_R))
    # cov=s_1_R*np.identity(n=R.shape[0])
    p_vec=discrete_gaussian_sample(dimension=m,th=1.5*math.sqrt(Sigma),cov=cov_p_yes.get())

    # sample0=cp.array(sample0)
    # sample10=cp.array(sample10)
    # print("------p_vec-----------\n")
    # p_vec=vector(ZZ,p_vec)
    # A=matrix(ZZ,A)
    # print("---??",A*p_vec)
    u_vec=cp.array(u_vec,dtype=int)
    p_vec=cp.array(p_vec,dtype=int)
    vec_v_0=cp.dot(A,p_vec)%q
    # print("-----------vec_v_0----------------\n",vec_v_0)
    vec_v=(u_vec-vec_v_0)%q
    # print("---------vec_v----------\n",vec_v)
    # print("?????",R.shape[0],R.shape[1])

    X_list=[]
    for u in vec_v:
        # print(u)
        # table0=table1establish(num=k+6,mode=0)
        # table1=table1establish(num=k+6, mode=1)
        x_vec=SampleX(k=k,u=u)
        X_list.append(x_vec)
    z_vec=cp.hstack(tuple(X_list))
    # z_vec=vector(ZZ,z_vec)
    # print("=================z_vec=================:\n",z_vec)
    x_vec_0=cp.dot(bias,z_vec)
    draw_proba(x_vec_0[0:R.shape[0]],name="x_vec0R")
    draw_proba(x_vec_0[R.shape[0]:x_vec_0.shape[0]],name="x_vec0I")
    
    # print("-----------x_vec_0-----------\n",x_vec_0)
    x_vec=cp.array(p_vec+x_vec_0,dtype=int)
    # bias_test_vec=cp.array(x_vec_0[0:R.shape[0]]+sample10)
    # draw_proba(bias_test_vec,name="bias_test_vec")
    # print("-----------x_vec---------------\n",list(x_vec))
    print("------{0}-------x-vec-length:------{1}---\n".format(cp.linalg.norm(x_vec),math.sqrt(Sigma)*math.sqrt(m)))
    print("===============\\beta=============",A.shape[1]**1.5)
    verify=cp.dot(A,x_vec)%q
    print("==================Verify===========\n")
    print(np.array_equal(verify.get(),u_vec.get()))
    # print(u_vec)
    e_vec=Derror(sigma=math.sqrt(Sigma),c=0,n=x_vec.shape[0],tau=2)
    draw_proba(x_vec[0:R.shape[0]],name="sampleR")
    draw_proba(x_vec[R.shape[0]:x_vec.shape[0]],name="sampleI")
    draw_proba(x_vec,name='sample')
    draw_proba(cp.array(e_vec),name='error')
    return x_vec,Sigma
def SamplePreObs(A_hat,A,u_vec,k,Bias,R,obs_list):
    m=A.shape[1]
    n=A.shape[0]
    q=2**k
    # qset=Zmod(q)
    s_1_G=4
    r=4.5
    m=A.shape[1]
    bias=Bias
    # cov_p_1=0.5*np.dot(bias,bias.T)
    # print("============cov_p_1=============:\n",cov_p_1)
    # print("======================================q==================:\n",q)
    # Sigma=(s_1_R**2+1)*(s_1_G+2)
    # s=svd(a=math.sqrt(0.5)*bias,compute_uv=False)
    # print("--------------------------sqrt(Sigma)------------------\n",s[0])
    SR=svd(a=R.get(),compute_uv=False)
    # print(type(s))
    s_1_R=SR[0]
    s_1_G=4
    r=4.5
    print("------------------s_1(R)--------------:\n",SR)
    Sigma=(s_1_R+6)**2
    # Sigma0=s_1_R*2
    # Sigma=math.sqrt(n*n*k)*(math.log2(n)**2)
    # Sigma=Sigma0**2
    # Sigma=2*cp.max(cp.diag(cov_p_1))+6
    # print("--------------Sigma0,Sigma-----------------\n",Sigma0,Sigma)
    print("-----------Sigma--------------------:\n",Sigma)


    # cov_p_yes=Sigma*cp.eye(N=cov_p_1.shape[0],M=cov_p_1.shape[1],dtype=int)-cov_p_1



    # test_cov_p=cov_p-cov_p_1
    # print("===================test_p_cov================\n",test_cov_p)
    # test_eigvals=np.linalg.eigvals(test_cov_p.get())
    # is_semidefinite=all(e>=0 for e in test_eigvals)
    # print("------------test_cov_p_ei-----------:\n",is_semidefinite)
    # is_semidefinite=all(e>=0 for e in test_eigvals)



    # print("--------cov_p-------------:\n",cov_p_yes)


    # eigvals, eigvectors=np.linalg.eig(cov_p.get())
    # print("------------cov_p_ei-----------:\n",eigvals)
    # is_semidefinite=all(e>=0 for e in eigvals)
    # print("----------------------cov_p_ei---------------------:\n",is_semidefinite)
    # sqrt_eigvals=np.sqrt(eigvals)
    # sqrt_cov_p=np.dot(eigvectors,np.dot(np.diag(sqrt_eigvals),eigvectors.T))
    # print("=================sqrt_cov_p=============================:\n",sqrt_cov_p)
    # print((r**2)*cov_p)
    # p_vec=discrete_gaussian_sample(mu= cp.zeros(shape=m),cov=cov_p,dimension=m,th=R.shape[1]/2.5)
    n=A.shape[0]
    # norm=np.max(np.linalg.norm(B,ord=2,axis=0))
    # sigma=norm*(math.sqrt(math.log(n))**1.36)
    # print("-----------sigmaInInIn-----------\n",sigma)
    # cov=s_1_R*np.identity(n=R.shape[0])
    p_vec,samvar=discrete_gaussian_sample_obs(k=k,n=n,sigma=math.sqrt(Sigma),R=R,bias=bias,obs_list=obs_list)

    # sample0=cp.array(sample0)
    # sample10=cp.array(sample10)
    # print("------p_vec-----------\n")
    # p_vec=vector(ZZ,p_vec)
    # A=matrix(ZZ,A)
    # print("---??",A*p_vec)
    u_vec=cp.array(u_vec,dtype=int)
    p_vec=cp.array(p_vec,dtype=int)
    vec_v_0=cp.dot(A,p_vec)%q
    # print("-----------vec_v_0----------------\n",vec_v_0)
    vec_v=(u_vec-vec_v_0)%q
    # print("---------vec_v----------\n",vec_v)
    # print("?????",R.shape[0],R.shape[1])

    X_list=[]
    for u in vec_v:
        # print(u)
        table0=table1establish(num=k+6,mode=0)
        table1=table1establish(num=k+6, mode=1)
        x_vec=SampleX(k=k,u=u)
        X_list.append(x_vec)
    z_vec=cp.hstack(tuple(X_list))
    # z_vec=vector(ZZ,z_vec)
    # print("=================z_vec=================:\n",z_vec)
    x_vec_0=cp.dot(bias,z_vec)
    draw_proba(x_vec_0[0:bias.shape[0]-bias.shape[1]],name="x_vec0R")
    draw_proba(x_vec_0[bias.shape[0]-bias.shape[1]:x_vec_0.shape[0]],name="x_vec0I")
    
    # print("-----------x_vec_0-----------\n",x_vec_0)
    x_vec=cp.array(p_vec+x_vec_0,dtype=int)
    # bias_test_vec=cp.array(x_vec_0[0:R.shape[0]]+sample10)
    # draw_proba(bias_test_vec,name="bias_test_vec")
    # print("-----------x_vec---------------\n",list(x_vec))
    print("------{0}-------x-vec-length:------{1}---\n".format(cp.linalg.norm(x_vec),math.sqrt(Sigma)*math.sqrt(m)))
    print("===============\\beta=============",A.shape[1]**1.5)
    verify=cp.dot(A,x_vec)%q
    print("==================Verify===========\n")
    print(np.array_equal(verify.get(),u_vec.get()))
    # print(u_vec)
    e_vec=Derror(sigma=math.sqrt(samvar),c=0,n=x_vec.shape[0],tau=3)
    draw_proba(x_vec[0:bias.shape[1]-bias.shape[0]],name="sampleR")
    draw_proba(x_vec[bias.shape[1]-bias.shape[0]:x_vec.shape[0]],name="sampleI")
    draw_proba(x_vec,name='sample')
    draw_proba(cp.array(e_vec),name='error')
    return x_vec,Sigma
def Uerror(size):
    return cp.array(np.random.randint(low=-1,high=1,size=size))
def Derror(c,sigma,n,tau):
    Der=DiscreteGaussianDistributionIntegerSampler(sigma=sigma,tau=tau,algorithm="uniform+table",c=c,precision="mp")
    Derror_list=[]
    while True:
        for i in range(n):
            Derror_list.append(Der())
        err_vec=np.array(Derror_list)
        norm=np.linalg.norm(ord=2,x=err_vec)
        print("---------------norm-------------",norm)
        print("------------normyes-------------",sigma*math.sqrt(n))
        if norm>sigma*math.sqrt(n):
            Derror_list.clear()
            continue
        else:
            break
    # print("====================Derror===================:\n",Derror_list)    
    # for i in range(100*100*100):
    #     Derror_list.append(Derror())
    # print("====================Derror===================:\n",Derror_list)
    return cp.array(err_vec)
def draw_proba(prob_vec,name="default"):

    # 定义数段区间
    bins = np.arange(-700, 700, 5)

    # 使用numpy.histogram计算每个区间的频数
    hist, bin_edges = np.histogram(prob_vec.get(), bins=bins)

    # 将频数转换为概率
    probabilities = hist / np.sum(hist)

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], probabilities, width=0.6, edgecolor='orange')
    plt.xlabel('Number Range')
    plt.ylabel('Probability')
    plt.title('Probability Distribution in Different Number Ranges')
    # plt.xticks(bin_edges)
    plt.savefig(name+'.pdf', transparent=True)

def encrypt(A,u,mbi,k,Sigma):
    q=2**k
    n=A.shape[0]
    m=A.shape[1]
    # print("---vec.T---\n",(cp.array(u)).T)
    s=np.random.randint(0,q,size=n,dtype=int)
    c_1_test=cp.dot(A.T,cp.array(s))%q
    derrorm=Derror(n=m,sigma=2*math.sqrt(n),c=0,tau=1)
    derrorm2=Derror(n=m,sigma=2*math.sqrt(n),c=0,tau=1)
    print("---------derrorm-------\n",derrorm)
    # uerror=Uerror(size=m)
    # c_1=(cp.dot(A.T,cp.array(s))+derrorm)%q
    c_1=(cp.dot(A.T,cp.array(s))+derrorm+derrorm2)%q
    derror1=Derror(n=1,sigma=2*math.sqrt(n),c=0,tau=1)
    # print("---------derror1-------\n",uerror)
    c_2=(mbi*(q/2)+cp.dot(cp.array(u),cp.array(s))+derror1[0])%q
    # uerror1=Uerror(size=1)
    # c_2=(mbi*(q/2)+cp.dot(cp.array(u),cp.array(s))+uerror1[0])%q
    print("-----------ciphertext-------\n",c_1,c_2)
    return c_1,c_2,(c_1_test,cp.dot(cp.array(u),cp.array(s))%q)
def decrypt(x_vec,c_1,c_2,k,test,mTrue):
    q=2**k
    mitest=cp.dot(x_vec,test[0])%q
    mi=(cp.dot(x_vec,c_1))%q
    print("---------------mi---------------\n",mitest)
    print("------------verify---------------\n",test[1])
    mi2=(c_2-mi)%q
    print("--------------------mi2-------------------\n",mi2)
    print("-----q/4-----\n",q/4)
    if mi2>=0 and mi2<q/4:
        m=0
    elif mi2>3*q/4 and mi2<q:
        m=0
    else:
        m=1
    print("-------------------------",m,mTrue)
    if m!=mTrue:
        raise SystemExit
if __name__=="__main__":
    # print(print_queue(table1establish(num=1000,mode=0)))
    k=24
    n=512
    q=2**k
    # qset=IntegerModRing(2**k)
    # table0=table1establish(num=1000,mode=0)
    # print(print_queue(table0))
    # table1=table1establish(num=1000, mode=1)
    # print(print_queue(table1))
    # u=generate_u(k=k)
    g_vec=generate_g(k=k)
    # x_vec=SampleX(k=k,u=u,table0=table0,table1=table1)
    # product_g_x=cp.array(g_vec)@cp.array(x_vec)
    # product_g_x=qset(product_g_x)
    # print(type(u))
    # print(u)
    # print(product_g_x)
    G=generate_G(g_vec=g_vec,n=n)
    S_k=generate_Sk(k=k)
    # print("g*Sk:\n",cp.dot(cp.array(g_vec),cp.array(S_k)).get()%q)
    S=generate_S(S_k=S_k,n=n)
    # A2,R2,A_hat2=TrapGen(n=n,k=k,G=G,S=None)
    A,A_hat,R,bias,obs_list=TrapGenObs(n=n,k=k,G=G,S=None)
    # A2,R2,A_hat2,BaseA2=TrapBaseGen(n=n,k=k,G=G,S=S)

    u_vec=np.random.randint(0,q,n,dtype=int)
    x_vec,Sigma=SamplePreObsPro(A_hat=A_hat,A=A,u_vec=u_vec,k=k,Bias=bias,R=R,obs_list=obs_list)

    # result=matrix(G)*matrix(S)
    # matrix(qset,result)
    # print("G*S:\n",matrix(Qset,G)*matrix(Qset,S))
    for i in range(128):
        m=np.random.randint(0,2)
        print("-------mTure-----------\n",m)
        c_1,c_2,test=encrypt(A=A,u=u_vec,k=k,Sigma=Sigma,mbi=m)
        decrypt(x_vec=x_vec,c_1=c_1,c_2=c_2,k=k,test=test,mTrue=m)

