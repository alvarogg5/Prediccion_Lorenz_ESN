import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
import random as rd
import math as mt
import pandas as pd


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
    #df = pd.read_csv('2006-2022a.csv')
    #u = df[df.columns[1:]].to_numpy()
    datos=pd.read_csv('2014-2022a.csv', usecols=range(1, 4))
    #print(datos)
    datos.drop(datos[datos['Chuvia'] == '-9,999.0'].index, inplace = True)
    datos.drop(datos[datos['Temperatura_med'] == '-9,999.00'].index, inplace = True)
    datos.drop(datos[datos['refacho_max'] == '-9,999.00'].index, inplace = True)
    #datos.drop(datos[datos['Chuvia'] == '0.0'].index, inplace = True)
    #datos['fecha']=pd.to_datetime(datos['fecha'])
    datos['Chuvia']=datos['Chuvia'].astype(float)
    datos['Temperatura_med']=datos['Temperatura_med'].astype(float)
    datos['refacho_max']=datos['refacho_max'].astype(float)
    print(datos)

    u =  datos.T
    u = u.values
    filas, columnas = u.shape
    print(u)

    print("Número de filas:", filas)
    print("Número de columnas:", columnas)
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
    print(f'Tikhonov training error: {norm(wout @ ra[:, :T] - u[:, :T]) / norm(u[:, :T])}')
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
    #lambda_max = 0.9056
    tim = tt
    #tim = tt / lambda_max * 0.02
    plt.figure()
    plt.plot(tim[:T-1], u[0,T+1:nT*T], tim[:T-1], u2[0,T+1:nT*T])
    #plt.title('True state and prediction (Lluvia)')
    plt.xlabel('$t$ (días)', fontsize=20)
    #plt.xlabel('$\lambda_{max}t$', fontsize=20)
    plt.ylabel('$Lluvia(t), Lluvia_R(t)$', fontsize=16)
    plt.legend(['$Lluvia(t)$', '$Lluvia_R(t)$'], fontsize=16)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()
    
    plt.figure()
    plt.plot(tim[:100], u[0,T+1:T+101], tim[:100], u2[0,T+1:T+101])
    #plt.title('True state and prediction (Lluvia)')
    plt.xlabel('$t$ (días)', fontsize=20)
    #plt.xlabel('$\lambda_{max}t$', fontsize=20)
    plt.ylabel('$Lluvia(t), Lluvia_R(t)$', fontsize=16)
    plt.legend(['$Lluvia(t)$', '$Lluvia_R(t)$'], fontsize=16)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()
    
    plt.figure()
    plt.plot(tim[:T-1], u[1,T+1:nT*T], tim[:T-1], u2[1,T+1:nT*T])
    #plt.title('True state and prediction (Racha máxima)')
    plt.xlabel('$t$ (días)', fontsize=20)
    #plt.xlabel('$\lambda_{max}t$', fontsize=20)
    plt.ylabel('$racha-max(t), racha-max_R(t)$', fontsize=16)
    plt.legend(['$racha-max(t)$', '$racha-max_R(t)$'], fontsize=16)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()
    
    plt.figure()
    plt.plot(tim[:100], u[1,T+1:T+101], tim[:100], u2[1,T+1:T+101])
    #plt.title('True state and prediction (Racha máxima)')
    plt.xlabel('$t$ (días)', fontsize=20)
    #plt.xlabel('$\lambda_{max}t$', fontsize=20)
    plt.ylabel('$racha-max(t), racha-max_R(t)$', fontsize=16)
    plt.legend(['$racha-max(t)$', '$racha-max_R(t)$'], fontsize=16)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()
    
    plt.figure()
    plt.plot(tim[:T-1], u[2,T+1:nT*T], tim[:T-1], u2[2,T+1:nT*T])
    #plt.title('True state and prediction (Temperatura media)',fontsize=20, fontweight='bold')
    plt.xlabel('$t$ (días)', fontsize=20)
    #plt.xlabel('$\lambda_{max}t$', fontsize=20)
    plt.ylabel('$temp-med(t), temp-med_R(t)$', fontsize=16)
    plt.legend(['$temp-med(t)$', '$temp-med_R(t)$'], fontsize=16)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()
    
    plt.figure()
    plt.plot(tim[:100], u[2,T+1:T+101], tim[:100], u2[2,T+1:T+101])
    #plt.title('True state and prediction (Temperatura media)',fontsize=20, fontweight='bold')
    plt.xlabel('$t$ (días)', fontsize=20)
    plt.ylabel('$temp-med(t), temp-med_R(t)$', fontsize=16)
    plt.legend(['$temp-med(t)$', '$temp-med_R(t)$'], fontsize=16)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()

    
    
    plt.figure()
    plt.plot(tim, norm_error, 'r')
    plt.title('Normalized error')
    plt.xlabel('$t$ (días)', fontsize=20)
#    plt.xlabel('$\lambda_{max}t$', fontsize=20)
    plt.ylabel('$E(t)$', fontsize=20)
    plt.show()
    
    ntt = np.arange(T+1, nT*T)
    total_norm_error = np.linalg.norm(u2[:,ntt] - u[:,ntt]) / np.linalg.norm(u[:,ntt])
    print(f'Tikhonov total pred. error: {total_norm_error}')
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(u[0,tt], u[1,tt], u[2,tt])
    ax.plot(u2[0,tt], u2[1,tt], u2[2,tt], 'r')
    plt.show()
    
    tt_v = np.where(norm_error < 0.4)[0]
    #print(f'Time: {tt_v}')
    t_v = len(tt_v)
    #t_v = len(tt_v) / lambda_max * 0.02
    #print(f'Usable time: {t_v}')
    

# Ejecutar             
repository_onlyL(M=3, T=730, nT=2, Dr=8000, d=4, rho=0.45, sigma=0.6, beta=0.01)

