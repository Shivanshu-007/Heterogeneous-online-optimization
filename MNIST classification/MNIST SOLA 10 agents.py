#!/usr/bin/env python
# coding: utf-8
# @Time     : 11/20/23
# @Author   : Shivanshu Tripathi, Darshan Gadginmath, Fabio Pasqualetti
# @FileName : Linear regression-DL.py# Simulation  setting: MNIST dataset with 10 agents
# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tt
from torchvision import datasets, transforms
from torch.utils import data
import pickle
import pdb
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# In[2]:


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a function to add Gaussian noise to an image
def add_gaussian_noise(image, mean, std):
    noise = torch.randn(image.size()) * std + mean
    noisy_image = image + noise
    return noisy_image

# Testing loop
def test(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_correct = 0
    test_total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            # Flatten the input images
            images = images.view(-1, 28 * 28)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Calculate average loss
    average_loss = test_loss / len(test_loader)

    # Calculate accuracy
    accuracy = 100 * test_correct / test_total

    return accuracy, average_loss


# In[3]:


# Parameters
num_agents = 10
num_epochs = 3000
cent_update_interval = 5
dist_update_interval = 5
storage_size = 128
line_topology = True
counter = 0
loss = 0.5
loss_cent = 0.5

# Create agent models and optimizers for distributed
models = [Net() for _ in range(num_agents)]
optimizers = [optim.SGD(model.parameters(), lr=0.01) for model in models]
# Model and optimizer for centralized
global_model = Net()
optimizer_cent = optim.SGD(global_model.parameters(), lr=0.01)

# Define the batch size
batch_size = 1


# In[4]:


# Centralized dataset
transform_with_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0, std=0.1))  # Adjust mean and std as needed
])
mnist_train_centdata = datasets.MNIST(root='./data', train=True, transform=transform_with_noise, download=True)
mnist_train_centloader = DataLoader(mnist_train_centdata, batch_size=batch_size, shuffle=True)

# Distributed dataset
transform_with_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0, std=5))  # Adjust mean and std as needed
])
mnist_train_distdata = datasets.MNIST(root='./data', train=True, transform=transform_with_noise, download=True)
mnist_train_distloader = DataLoader(mnist_train_distdata, batch_size=batch_size, shuffle=True)

# Create nine subsets for digits 0-9
subset_0_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 0)])
subset_1_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 1)])
subset_2_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 2)])
subset_3_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 3)])
subset_4_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 4)])
subset_5_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 5)])
subset_6_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 6)])
subset_7_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 7)])
subset_8_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 8)])
subset_9_indices = torch.tensor([i for i, label in enumerate(mnist_train_distdata.targets) if (label == 9)])

subset_0_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_0_indices)
subset_1_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_1_indices)
subset_2_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_2_indices)
subset_3_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_3_indices)
subset_4_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_4_indices)
subset_5_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_5_indices)
subset_6_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_6_indices)
subset_7_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_7_indices)
subset_8_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_8_indices)
subset_9_dataset = torch.utils.data.Subset(mnist_train_distdata, subset_9_indices)


mnist_train_distdatas = []

mnist_train_distdatas.append(subset_0_dataset)
mnist_train_distdatas.append(subset_1_dataset)
mnist_train_distdatas.append(subset_2_dataset)
mnist_train_distdatas.append(subset_3_dataset)
mnist_train_distdatas.append(subset_4_dataset)
mnist_train_distdatas.append(subset_5_dataset)
mnist_train_distdatas.append(subset_6_dataset)
mnist_train_distdatas.append(subset_7_dataset)
mnist_train_distdatas.append(subset_8_dataset)
mnist_train_distdatas.append(subset_9_dataset)

# Define a transform to preprocess the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_datasetLoader = torch.utils.data.DataLoader(mnist_test_dataset,batch_size=batch_size,shuffle=False)


# In[5]:


average_train_loss_values = []
average_train_accuracy_values = []
average_loss_values = []
average_accuracy_values = []


# In[6]:


import pdb
data_storage_cent = []
data_storage_dist = []
test_accuracies = []
test_losses = []
dist_model = Net()
cent_model = Net()
cent_comm = True
for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    print ('epoch = ', epoch)
    if cent_comm:    
        print('Cent')
        for i, (images, labels) in enumerate(mnist_train_centloader):
            image, label = images[0], labels[0]
            if len(data_storage_cent) < storage_size:          # Add to data storage
                data_storage_cent.append((image, label))
            else:  
                data_storage_cent[counter % storage_size] = (image, label)
                
            data_batch = torch.stack([item[0].view(-1, 28 * 28) for item in data_storage_cent])
            label_batch = torch.LongTensor([item[1] for item in data_storage_cent])
            data_batch = data_batch.view(-1, 28 * 28)
            optimizer_cent.zero_grad()             # Zero the gradients            
            outputs = global_model(data_batch)    # Forward pass
            loss_cent = nn.CrossEntropyLoss()(outputs, label_batch)     
            _, train_predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (train_predicted == labels).sum().item()
            if counter > 0:            # Save the current model parameters for the next iteration
                for current_param, dist_param in zip(global_model.parameters(), dist_model.parameters()):
                    current_param.data = 1/(1/loss_cent+1/loss)*(1/loss_cent * current_param.data + 1/loss * dist_param.data)
            loss_cent.backward()       
            optimizer_cent.step()
            cent_model.load_state_dict(global_model.state_dict()) 
            break

            
        average_train_loss = loss_cent
        average_train_accuracy = 100 * train_correct / train_total
        average_train_loss_values.append(average_train_loss)
        average_train_accuracy_values.append(average_train_accuracy)
        
        average_weights = {}
        for param_name in global_model.state_dict():
            average_weights[param_name] = torch.zeros_like(global_model.state_dict()[param_name])
            average_weights[param_name] += global_model.state_dict()[param_name]

        # Set the global model's weights to the average weights
        global_model.load_state_dict(average_weights)
        counter += 1            # Update the counter
        if counter % cent_update_interval == 0:
            for agent_id in range(num_agents):
                models[agent_id].load_state_dict(global_model.state_dict(average_weights))              
            cent_comm = False
            counter = 0
            
    else:
        print('Dist')
        for agent_id in range(num_agents):
            model = models[agent_id]
            optimizer = optimizers[agent_id]
            train_loader = data.DataLoader(mnist_train_distdatas[agent_id], batch_size=batch_size, shuffle=True)

            data_storage_dist = [[] for _ in range(len(train_loader))]
            for batch_idx, (images, labels) in enumerate(train_loader):
                image, label = images[0], labels[0]

                # Add to data storage
                if len(data_storage_dist[batch_idx]) < storage_size:
                    data_storage_dist[batch_idx].append((image, label))
                else:
                    data_storage_dist[batch_idx][counter % storage_size] = (image, label)

                # Convert data storage to a batch for training
                data_batch = torch.stack([item[0].view(-1, 28 * 28) for item in data_storage_dist[batch_idx]])
                label_batch = torch.LongTensor([item[1] for item in data_storage_dist[batch_idx]])

                # Flatten the input images
                data_batch = data_batch.view(-1, 28 * 28)
                optimizer.zero_grad()
                outputs = model(data_batch)

                # Compute loss
                loss = nn.CrossEntropyLoss()(outputs, label_batch)

                _, train_predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (train_predicted == labels).sum().item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                ## Considering line communication
                # Communicate with right neighbor
                right_neighbor_id = (agent_id + 1) % num_agents
                for param_name, param in model.named_parameters():
                    param.data = 0.5 * (param.data + models[right_neighbor_id].state_dict()[param_name])

                # Communicate with left neighbor
                left_neighbor_id = (agent_id - 1) % num_agents
                for param_name, param in model.named_parameters():
                    param.data = 0.5 * (param.data + models[left_neighbor_id].state_dict()[param_name])
            break        

        average_train_loss = loss/len(train_loader)
        average_train_accuracy = 100 * train_correct / train_total

        average_train_loss_values.append(average_train_loss)
        average_train_accuracy_values.append(average_train_accuracy)
            
        avg_params = {}        # Compute the average parameter value of all models
        for param_name, param in model.named_parameters():
            param_sum = torch.zeros_like(param.data)            # Initialize a tensor to store the sum of parameters

            for agent_model in models:            # Sum up the parameter values across all models
                param_sum += agent_model.state_dict()[param_name]

            avg_params[param_name] = param_sum / num_agents            # Calculate the average by dividing by the number of agents

        global_model.load_state_dict(avg_params)        # Update the global model with the average parameter values
        
        dist_model.load_state_dict(global_model.state_dict())        # Save the current model parameters for the next iteration
        

        if counter > 0:        # Save the current model parameters for the next iteration
            for current_param, cent_param in zip(global_model.parameters(), cent_model.parameters()):
                current_param.data = (1/(1/loss_cent+1/loss))*(1/loss * current_param.data + 1/loss_cent * cent_param.data)

        # Update the counter
        counter += 1
        
        if counter % dist_update_interval == 0:
            cent_comm = True
            counter = 0        
    
    criterion = nn.CrossEntropyLoss()
    test_accuracy, test_loss = test(global_model, test_datasetLoader, criterion)
    print(f'Test Accuracy: {test_accuracy}%')
    print(f'Test Loss: {test_loss}')
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)


# In[8]:


# Plot the loss and accuracy curves for different num_agents
plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.plot(test_accuracies, label=f'Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 2)
plt.plot(test_losses, label=f'Average loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

