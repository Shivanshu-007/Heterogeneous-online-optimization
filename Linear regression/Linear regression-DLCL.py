#!/usr/bin/env python
# coding: utf-8
# @Time     : 11/20/23
# @Author   : Shivanshu Tripathi, Darshan Gadginmath, Fabio Pasqualetti
# @FileName : Linear regression-DLCL.py# Simulation  setting: Linear regression with switching between distributed learning and centralized learning
# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt


n = 3
p = 1
D = 50000

true_beta = np.array([1, 3, 5])

eta = 0.01
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


# In[3]:


T = 3000  # Total number of iterations
N = 10  # Number of agents
W = generate_stochastic_matrix(N)-np.eye(N)

# Initialize matrices
datadist = T
DataAgentsX = np.zeros((n, datadist, N))
DataAgentsY = np.zeros((p, datadist, N))

dataser = np.arange(1, T + 1)
for i in range(N):
    samples = np.random.choice(dataser, datadist)
    DataAgentsX[:, :, i] = xdata[:, samples - 1] 
    DataAgentsY[:, :, i] = ydata[:, samples - 1] 

# Initialize beta, beta_naive
beta = np.random.randn(p, n)
beta_naive = np.copy(beta)

switchS1 = 50
switchS2 = 50

count = 0
state = 1

# S1 initializations
Ws1 = np.eye(p)
ts1 = np.trace(Ws1)

betaS1 = np.copy(beta)
betaS1_n = np.copy(beta)

s1count = 0

# S2 initializations
Ws2 = np.eye(p)
ts2 = np.trace(Ws2)

betaS2 = np.copy(beta)
betaS2_n = np.copy(beta)

betaS2_dist = np.tile(beta, (1, 1, N))
betaS2_dist_n = np.tile(beta, (1, 1, N))

s2count = 0

# Initialize arrays to keep track of beta for plots
beta_traj = np.zeros((p, n, T))
beta_n_traj = np.zeros((p, n, T))
betaNorm = np.zeros((1, T))
betaNorm_naive = np.zeros((1, T))


# In[4]:


for i in range(1, T):
    count += 1
    
    if state == 1:
        # Centralized
        s1count += 1

        if count == switchS1:
            state = 2
            count = 0

        DataS1X = CentAgentX[:, :s1count]
        DataS1Y = CentAgentY[:, :s1count]

        # Subsystem 1
        gradS1 = gradientlinear(DataS1Y, DataS1X, betaS1, n, p, s1count)
        betaS1 = beta - eta * gradS1

        errS1 = DataS1Y - np.dot(betaS1, DataS1X)
        Ws1 = np.linalg.inv(1 / s1count * np.dot(errS1, errS1.T))
        ts1 = np.trace(Ws1)

        beta = 1 / (ts1 + ts2) * (ts1 * betaS1 + ts2 * betaS2)
        beta_traj[:, :, i] = beta

        # Naive S1
        gradS1_n = gradientlinear(DataS1Y, DataS1X, betaS1_n, n, p, s1count)
        betaS1_n = beta_naive - eta * gradS1_n
        beta_naive = betaS1_n

        beta_n_traj[:, :, i] = beta_naive
        betaNorm[:, i] = np.linalg.norm(beta - true_beta)
        betaNorm_naive[:, i] = np.linalg.norm(beta_naive - true_beta)

    elif state == 2:
        s2count += 1

        if count == switchS2:
            state = 1
            count = 0

        # Decentralized learning
        DataS2X = DataAgentsX[:, :s2count, :]
        DataS2Y = DataAgentsY[:, :s2count, :]

        betaS2_temp = np.zeros((p, n, N))

        for j in range(N):
            gradN = gradientlinear(DataS2Y[:, :, j], DataS2X[:, :, j], beta, n, p, s2count)
            betaS2_temp[:, :, j] = betaS2 - eta * gradN

        betaS2_dist = np.zeros((p, n, N))

        for j in range(N):
            for k in range(N):
                betaS2_dist[:, :, j] += W[j, k] * betaS2_temp[:, :, k]

        betaS2 = np.mean(betaS2_dist, axis=2)

        errS2 = np.zeros(DataS2Y[:, :, 0].shape)

        for j in range(N):
            errS2 += DataS2Y[:, :, j] - np.dot(betaS2, DataS2X[:, :, j])

        Ws2 = np.linalg.inv(1 / s2count * np.dot(errS2, errS2.T))
        ts2 = np.trace(Ws2)

        beta = 1 / (ts1 + ts2) * (ts1 * betaS1 + ts2 * betaS2)
        beta_traj[:, :, i] = beta

        # Naive distributed updates
        betaS2_temp_n = np.zeros((p, n, N))

        for j in range(N):
            gradN_n = gradientlinear(DataS2Y[:, :, j], DataS2X[:, :, j], beta_naive, n, p, s2count)
            betaS2_temp_n[:, :, j] = beta_naive - eta * gradN_n

        betaS2_dist_n = np.zeros((p, n, N))

        for j in range(N):
            for k in range(N):
                betaS2_dist_n[:, :, j] += W[j, k] * betaS2_temp_n[:, :, k]

        betaS2_n = np.mean(betaS2_dist_n, axis=2)
        beta_naive = betaS2_n

        beta_n_traj[:, :, i] = beta_naive

        betaNorm[:, i] = np.linalg.norm(beta - true_beta)
        betaNorm_naive[:, i] = np.linalg.norm(beta_naive - true_beta)


# In[5]:


num_iterations = betaNorm.shape[1]  # Assuming the number of iterations is along axis 1

plt.figure(figsize=(10, 6))
for i in range(betaNorm.shape[0]):
    plt.plot(np.arange(num_iterations), betaNorm_naive[i], label=f'Beta Norm Naive {i}')

plt.xlabel('Epochs')
plt.ylabel('Norm Value (beta norm naive)')
plt.legend()
plt.grid(True)
plt.show()

num_iterations = betaNorm.shape[1]  # Assuming the number of iterations is along axis 1

plt.figure(figsize=(10, 6))
for i in range(betaNorm.shape[0]):
    plt.plot(np.arange(num_iterations), betaNorm[i], label=f'Beta Norm {i}')

plt.xlabel('Epochs')
plt.ylabel('Norm Value (beta norm)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




