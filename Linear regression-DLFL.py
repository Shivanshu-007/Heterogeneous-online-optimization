#!/usr/bin/env python
# coding: utf-8
# @Time     : 11/20/23
# @Author   : Shivanshu Tripathi, Darshan Gadginmath, Fabio Pasqualetti
# @FileName : Linear regression-DLFL.py# Simulation  setting: Linear regression with switching between distributed learning and federated learning
# In[1]:


import numpy as np
import random

n = 3
p = 1
D = 50000

true_beta = np.array([1, 3, 5])

eta = 0.001
datanoise_cov = 30

xdata = np.random.randn(n, D) * 5

ydata = np.dot(true_beta, xdata) + np.random.randn(p, D) * datanoise_cov

# Central data
CentD = D
xcovcent = 2
centnoise = 3

CentAgentX = np.random.randn(n, CentD) * xcovcent
CentAgentY = np.dot(true_beta, CentAgentX) + np.random.randn(p, CentD) * centnoise


# In[2]:


def generate_stochastic_matrix(n):
    c = np.abs(np.random.randn(n))
    csum = np.sum(c)
    c = c / csum

    M2 = np.eye(n)

    for k in range(n):
        idx = np.random.permutation(n)
        idx = idx + np.arange(n) * n
        M2.flat[idx] = M2.flat[idx] + c[k]

    return M2

def gradientlinear(ydata, xdata, beta0, n, p, D):
    dbeta = np.zeros((p, n))

    for j in range(D):
        dbeta += np.outer(ydata[:, j] - np.dot(beta0, xdata[:, j]), xdata[:, j])

    dbeta = -dbeta / D
    return dbeta

def cost_cal(ydata, xdata, beta0, D):
    cost = 0
    for j in range(D):
        cost += np.sum((ydata[:, j] - beta0 * xdata[:, j])**2)
    
    cost = np.sum(cost) / D
    return cost


# In[3]:


T = 3000
N = 5
W = generate_stochastic_matrix(N)-np.eye(N)

datadist = T

DataAgentsX = np.zeros((n, datadist, N))
DataAgentsY = np.zeros((p, datadist, N))
FLAgentsX = np.zeros((n, datadist, N))
FLAgentsY = np.zeros((p, datadist, N))

beta = np.random.randn(p, n)
beta_naive = beta

switchS1 = 15
switchS2 = 200
count = 0
state = 1

# S1 initializations
Ws1 = np.eye(p)
ts1 = np.trace(Ws1)
betaS1 = beta
betaS1_n = beta
betaS1_dist = np.tile(beta[:, :, np.newaxis], (1, 1, N))
betaS1_dist_n = np.tile(beta[:, :, np.newaxis], (1, 1, N))
s1count = 1

# S2 initializations
Ws2 = np.eye(p)
ts2 = np.trace(Ws2)
betaS2 = beta
betaS2_n = beta
betaS2_dist = np.tile(beta[:, :, np.newaxis], (1, 1, N))
betaS2_dist_n = np.tile(beta[:, :, np.newaxis], (1, 1, N))
s2count = 1

# keep track of betas for plots
beta_traj = np.zeros((p, n, T))
beta_traj[:, :, 0] = beta
beta_n_traj = np.zeros((p, n, T))
beta_n_traj[:, :, 0] = beta_naive

betaNorm = np.zeros(T)
betaNorm_naive = np.zeros(T)

dataser = np.arange(1, T + 1)  # Generating a numpy array from 1 to T
for i in range(N):
    samples = random.sample(dataser.tolist(), datadist)
    
    DataAgentsX[:, :, i] = xdata[:, samples]
    DataAgentsY[:, :, i] = ydata[:, samples]

dataser = np.arange(1, T + 1)  # Generating a numpy array from 1 to T
for i in range(N):
    samples = random.sample(dataser.tolist(), datadist)
    
    FLAgentsX[:, :, i] = CentAgentX[:, samples]
    FLAgentsY[:, :, i] = CentAgentY[:, samples]


# In[4]:


for i in range(1, T + 1):
    count += 1
    
    if state == 1:
        s1count += 1

        if count == switchS1:
            state = 2
            count = 0

        DataS1X = FLAgentsX[:, :s1count, :]
        DataS1Y = FLAgentsY[:, :s1count, :]
        
        # Subsystem 1
        
        numeles = 3
        choice = np.random.choice(N, numeles, replace=False)
        
        for j in range(N):
            if np.isin(j, choice):
                gradN = gradientlinear(DataS1Y[:, :, j], DataS1X[:, :, j], beta, n, p, s1count)
                betaS1_dist[:, :, j] = beta - eta * gradN

        limbeta = betaS1_dist[:, :, choice]
        betaS1 = np.mean(limbeta, axis=2)

        errS1 = np.zeros(DataS1Y[:, :, 0].shape)
        
        for j in range(N):
            if np.isin(j, choice):
                errS1 += DataS1Y[:, :, j] - betaS1 @ DataS1X[:, :, j]

        errS1 /= numeles
        Ws1 = np.linalg.inv(1 / s1count * (errS1 @ errS1.T))
        ts1 = np.trace(Ws1)

        err_prev = np.zeros(DataS1Y[:, :, 0].shape)

        for j in range(N):
            if np.isin(j, choice):
                err_prev += DataS1Y[:, :, j] - beta @ DataS1X[:, :, j]

        err_prev /= numeles
        Ws_prev = np.linalg.inv(1 / s1count * (err_prev @ err_prev.T))
        ts_prev = np.trace(Ws_prev)
        
        alpha = ts1 / (ts_prev + ts1)
        beta = alpha * betaS1 + (1 - alpha) * beta

        beta_traj[:, :, i - 1] = beta
        
#         # Naive distributed updates
        
        betaS1_temp_n = betaS1_dist_n.copy()
        
        for j in range(N):
            if np.isin(j, choice):
                gradN = gradientlinear(DataS1Y[:, :, j], DataS1X[:, :, j], beta, n, p, s1count)
                betaS1_temp_n[:, :, j] = beta_naive - eta * gradN

        betaS1_dist_n = np.zeros((p, n, N))
        
        for j in range(N):
            for k in range(N):
                betaS1_dist_n[:, :, j] += W[j, k] * betaS1_temp_n[:, :, k]

        betaS1_n = np.mean(betaS1_dist_n, axis=2)
        beta_naive = betaS1_n

        beta_n_traj[:, :, i - 1] = beta_naive

    elif state == 2:
        s2count += 1

        if count == switchS2:
            state = 1
            count = 0

        # decentralized learning
        DataS2X = DataAgentsX[:, :s2count, :]
        DataS2Y = DataAgentsY[:, :s2count, :]
        
        betaS2_temp = betaS2_dist.copy()
        
        for j in range(N):
            gradN = gradientlinear(DataS2Y[:, :, j], DataS2X[:, :, j], beta, n, p, s2count)
            betaS2_temp[:, :, j] = beta - eta * gradN

        betaS2_dist = np.zeros((p, n, N))
        
        for j in range(N):
            for k in range(N):
                betaS2_dist[:, :, j] += W[j, k] * betaS2_temp[:, :, k]

        betaS2 = np.mean(betaS2_dist, axis=2)

        errS2 = np.zeros(DataS2Y[:, :, 0].shape)
        
        for j in range(N):
            errS2 += DataS2Y[:, :, j] - betaS2 @ DataS2X[:, :, j]

        errS2 /= N
        Ws2 = np.linalg.inv(1 / s2count * (errS2 @ errS2.T))
        ts2 = np.trace(Ws2)

        err_prev = np.zeros(DataS2Y[:, :, 0].shape)

        for j in range(N):
            err_prev += DataS2Y[:, :, j] - beta @ DataS2X[:, :, j]

        err_prev /= N
        Ws_prev = np.linalg.inv(1 / s2count * (err_prev @ err_prev.T))
        ts_prev = np.trace(Ws_prev)
        
        alpha = ts2 / (ts_prev + ts2)
        beta = (1 - alpha) * beta + alpha * betaS2
        
        beta_traj[:, :, i - 1] = beta
        
        # Naive distributed updates
        
        betaS2_temp_n = betaS2_dist_n.copy()
        
        for j in range(N):
            gradN = gradientlinear(DataS2Y[:, :, j], DataS2X[:, :, j], beta_naive, n, p, s2count)
            betaS2_temp_n[:, :, j] = beta_naive - eta * gradN

        betaS2_dist_n = np.zeros((p, n, N))
        
        for j in range(N):
            for k in range(N):
                betaS2_dist_n[:, :, j] += W[j, k] * betaS2_temp_n[:, :, k]

        betaS2_n = np.mean(betaS2_dist_n, axis=2)
        beta_naive = betaS2_n

        beta_n_traj[:, :, i - 1] = beta_naive
        
    betaNorm[i - 1] = np.linalg.norm(beta - true_beta)
    betaNorm_naive[i - 1] = np.linalg.norm(beta_naive - true_beta)


# In[15]:


import matplotlib.pyplot as plt

# Assuming betaNorm and betaNorm_naive are populated with values
# Plotting betaNorm
plt.plot(range(1, T + 1), betaNorm, 'r-', linewidth=1.6, label='switched algo')
plt.plot(range(1, T + 1), betaNorm_naive, 'b-', linewidth=1.6, label='naive algo')

plt.legend()
plt.xlabel('Iterations')
plt.ylabel(r'$||\beta-\beta^*||$')
plt.title('Norm vs Iterations')

plt.yscale('log')

plt.show()


# In[10]:


print(betaNorm_naive.shape)


# In[11]:


betaNorm = betaNorm.tolist()
betaNorm_naive = betaNorm_naive.tolist()


# In[14]:


# Assuming betaNorm_list is the Python list you want to write to a text file

# File path where you want to save the text file
file_path = "/Users/shivanshutripathi/Desktop/Project matlab codes/Linear reg files/DLFL/betaNorm_naive.txt"

# Writing the list elements along with iteration number to a text file
with open(file_path, 'w') as file:
    for i, value in enumerate(betaNorm_naive, start=1):
        file.write(f"{value},{i}\n")


# In[ ]:




