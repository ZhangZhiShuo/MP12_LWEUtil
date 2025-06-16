# 这个代码库 实现了MP12论文中的格上的核心算法，可以作为构造其他基于LWE和SVP及其变种困难问题密码方案的基础模块。该代码由电子科技大学 信息与软件工程学院 网络空间安全实验室 张智烁博士及廖永建教授实现、调试并开源。
# This code repository implements the core lattice-based algorithms from the MP12 paper, serving as fundamental building blocks for constructing cryptographic schemes based on hard problems like LWE (Learning With Errors) and SVP (Shortest Vector Problem) and their variants. The code was developed, debugged, and open-sourced by Dr. Zhang Zhishuo and Professor Liao Yongjian from the Cyberspace Security Laboratory, School of Information and Software Engineering, University of Electronic Science and Technology of China.
import numpy as np
import cupy as cp
from MP12_TrapdoorGen import *
from MP12_SamplePre import *
import math
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