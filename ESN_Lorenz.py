import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
import random as rd
import math as mt


# Generate A
def A_build(width, p, rho):
    """Definition of the A matrix"""
    width = int(width)
    A = np.zeros((width, width))
    for i in range(width):
        for j in range(i+1, width):
            if rd.random() <= p:
                A[i,j] = rd.uniform(-1, 1) #-1+2*rd.random()
                A[j,i] = A[i,j]
    return A*rho/np.real(max(np.linalg.eig(A)[0]))

# Generate W_in
def win_build(width, height, sigma):
    """Definition of the input matrix."""
    width = int(width)
    height = int(height)
    W = np.zeros((width, height))
    for i in range(width):
        x = int(mt.ceil(rd.uniform(0, height)))-1#rd.random()*height))-1
        #sigma2 = 2*sigma
        W[i,x] = rd.uniform(-sigma, sigma)#-sigma + sigma2*rd.random()
    return W

def repository_onlyL(M, T, nT, Dr, d, rho, sigma, beta):
# Parameters
    # Dr=500;
    # M=3;
    # d=3;
    # rho=0.4;
    # sigma=0.15;
    # T=500;
    # nT=2;
    # beta=0.01;
    p = d / Dr;
    
    # Generate u
    u = np.loadtxt('u_csv10000.csv', delimiter=',')
    # W_in and A
    win = win_build(Dr, M, sigma)
    A = A_build(Dr,p,rho)
    # Initialize   
    r = np.zeros((Dr, nT*T))
    ra = np.copy(r)
    # First training to obtain W_out
    for i in range(1, T):
        r[:,i] = np.tanh(np.dot(A, r[:, i-1])+np.dot(win, u[:, i-1]))
        ra[:, i] = r[:, i]
        ra[1::2,i] = r[1::2, i]**2
    # Create output matrix
    prod = np.dot(u[:,0:T],np.transpose(ra[:,0:T]))
    inv = np.linalg.inv(np.dot(ra[:,0:T],np.transpose(ra[:,0:T]))+beta*np.eye(Dr))
    wout = np.dot(prod, inv)
    #print(f'Tikhonov training error: {norm(wout @ ra[:, :T] - u[:, :T]) / norm(u[:, :T])}')
    # Initialize prediction
    r = np.zeros((Dr, nT*T))
    ra = np.copy(r)
    u2 = np.zeros((M, nT*T))
    u2[:,0] = u[:,0]
    #Small training to start the prediction
    for i in range(1,T):
        r[:,i] = np.tanh(np.dot(A, r[:, i-1])+np.dot(win, u[:, i-1]))
        ra[:, i] = r[:, i]
        ra[1::2,i] = r[1::2, i]**2
        u2[:,i] = np.dot(wout,ra[:,i])
    #Prediction
    norm_error = np.zeros(nT*T - T)
    for i in range(T, nT*T):
        r[:,i] = np.tanh(np.dot(A, r[:, i-1])+np.dot(win, u2[:, i-1]))
        ra[:, i] = r[:, i]
        ra[1::2,i] = r[1::2, i]**2
        u2[:,i] = np.dot(wout,ra[:,i])
        norm_error[i - T - 1] = np.linalg.norm(u2[:,i] - u[:,i]) / np.linalg.norm(u[:,i])
    
    
    
        
    # Plot
    tt = np.arange(1, (nT-1)*T +1)
    lambda_max = 0.9056
    tim = tt / lambda_max * 0.02
    plt.figure()
    plt.plot(tim[:T-1], u[1,T+1:nT*T], tim[:T-1], u2[1,T+1:nT*T])
    plt.title('True state and prediction')
    plt.xlabel('$\lambda_{max}t$', fontsize=20)
    plt.ylabel('$x(t), x_R(t)$', fontsize=20)
    plt.legend(['$x(t)$', '$x_R(t)$'], fontsize=20)
    plt.show()
    
    plt.figure()
    plt.plot(tim, norm_error, 'r')
    plt.title('Normalized error')
    plt.xlabel('$\lambda_{max}t$', fontsize=20)
    plt.ylabel('$E(t)$', fontsize=20)
    plt.show()
    
    ntt = np.arange(T+1, nT*T)
    total_norm_error = np.linalg.norm(u2[:,ntt] - u[:,ntt]) / np.linalg.norm(u[:,ntt])
    print(f'Tikhonov total pred. error: {total_norm_error}')
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(u[0,tt], u[1,tt], u[2,tt], label='Real')
    ax.plot(u2[0,tt], u2[1,tt], u2[2,tt], 'r', label='Predicción')
    # Agregar leyenda
    ax.legend()
    # Etiquetas de los ejes
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('y', fontweight='bold')
    ax.set_zlabel('z', fontweight='bold')

    plt.show()
    tt_v = np.where(norm_error < 0.4)[0]
    #print(f'Time: {tt_v}')
    t_v = len(tt_v) / lambda_max * 0.02
    print(f'Usable time: {t_v}')
    return t_v


# ds = np.arange(100, 1600, 100)
# t_vs = []

# for Dr in ds:
#     t_v = repository_onlyL(M=3, T=1000, nT=2, Dr=Dr, d=3, rho=0.45, sigma=0.2, beta=1.4)
#     t_vs.append(t_v)

# plt.plot(ds, t_vs)
# plt.xlabel('rho')
# plt.ylabel('t_v')
# plt.title('Usable Time vs. rho')
# plt.show()

repository_onlyL(M=3, T=1000, nT=2, Dr=800, d=3, rho=0.45, sigma=0.2, beta=1.4)


