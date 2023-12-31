#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tt
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn.functional as F
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
num_epochs = 1000
cent_update_interval = 5
dist_update_interval = 5
storage_size = 128
num_classes=10
counter = 0
loss = 0.5
loss_cent = 0.5
cent_comm = True

# Create agent models and optimizers
models = [Net() for _ in range(num_agents)]
optimizers = [optim.SGD(model.parameters(), lr=0.01) for model in models]
# Model and optimizer for centralized
global_model = Net()
optimizer_cent = optim.SGD(global_model.parameters(), lr=0.01)

# Define the batch size
batch_size = 1  # You need to set an appropriate batch size

# Some intermediate models
dist_model = Net()
cent_model = Net()


# In[4]:


# Centralized dataset
transform_with_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0, std=0.1))  # Adjust mean and std as needed
])
mnist_train_centdata = datasets.MNIST(root='./data', train=True, transform=transform_with_noise, download=True)
mnist_train_centloader = DataLoader(mnist_train_centdata, batch_size=batch_size, shuffle=True)

# image, label = mnist_train_centdata[0]

# image = image.view(28, 28)

# # Display the image using imshow()
# plt.imshow(image, cmap='gray')
# plt.title(f"Label: {label}")
# plt.show()

# pdb.set_trace()

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


# Creating lists to store parameters
average_train_loss_values = []
average_train_accuracy_values = []
average_loss_values = []
average_accuracy_values = []
data_storage_cent = []
data_storage_dist = []
test_accuracies = []
test_losses = []


# In[ ]:


## Training and testing using the proposed algorithm
for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    
    if cent_comm:
        for i, (images, labels) in enumerate(mnist_train_centloader):
            image, label = images[0], labels[0]

            # Storage size of 128 data samples for the data.
            if len(data_storage_cent) < storage_size:
                data_storage_cent.append((image, label))
            else:
                data_storage_cent[counter % storage_size] = (image, label)  # Replace the oldest data point

            # Prepare data and labels
            full_data_batch = torch.stack([item[0].view(-1, 28 * 28) for item in data_storage_cent])
            full_label_batch = torch.LongTensor([item[1] for item in data_storage_cent])

            optimizer_cent.zero_grad()
            outputs = global_model(full_data_batch)
            outputs = outputs.squeeze(1) 

            # Calculate loss
            loss_cent = nn.CrossEntropyLoss()(outputs, full_label_batch)  # Use class indices for CrossEntropyLoss

            # Applying proposed algorithm (Update model parameters)
            if counter > 0:
                for current_param, dist_param in zip(global_model.parameters(), dist_model.parameters()):
                    current_param.data = 1 / (1 / loss_cent + 1 / loss) * (1 / loss_cent * current_param.data + 1 / loss * dist_param.data)

            # Backward pass and optimization
            loss_cent.backward()
            optimizer_cent.step()
            cent_model.load_state_dict(global_model.state_dict())  # Save the current model parameters for the distributed system

            _, train_predicted = torch.max(outputs.data, 1)
            train_total += full_label_batch.size(0)  # Use original label size
            train_correct += (train_predicted == full_label_batch).sum().item()  # Compare with original labels

            average_train_loss = loss_cent.item()  # Get the loss value for this iteration
            average_train_accuracy = 100 * train_correct / len(data_storage_cent)
            average_train_loss_values.append(average_train_loss)
            average_train_accuracy_values.append(average_train_accuracy)
            break

        
        # Average of parameters to update the global model
        average_weights = {}
        for param_name in global_model.state_dict():
            average_weights[param_name] = torch.zeros_like(global_model.state_dict()[param_name])
            average_weights[param_name] += global_model.state_dict()[param_name]
        global_model.load_state_dict(average_weights)
        # Update the counter
        counter += 1            
        if counter % cent_update_interval == 0:
            # Initialize models for all agents with the parameters of the last global model
            for agent_id in range(num_agents):
                models[agent_id].load_state_dict(global_model.state_dict(average_weights))               
            cent_comm = False
            counter = 0
            
    else:
        # Training loop for distributed dataset
        loss = 0
        average_train_accuracy = 0
        for agent_id in range(num_agents):
            train_correct = 0
            train_total = 0
            model = models[agent_id]
            optimizer = optimizers[agent_id]
            train_loader = data.DataLoader(mnist_train_distdatas[agent_id], batch_size=batch_size, shuffle=True)

            # Initialize storage for last 128 data points for each agent
            data_storage_dist = [[] for _ in range(len(train_loader))]
            for batch_idx, (images, labels) in enumerate(train_loader):
                image, label = images[0], labels[0]
                # Storage size of 128 data samples for the data.
                if len(data_storage_dist[batch_idx]) < storage_size:
                    data_storage_dist[batch_idx].append((image, label))
                else:
                    data_storage_dist[batch_idx][counter % storage_size] = (image, label) # Replace the oldest data point

                # Calculate the loss using the entire dataset or a fixed subset
                full_data_batch_dist = torch.stack([item[0].view(-1, 28 * 28) for item in data_storage_dist[batch_idx]])
                full_label_batch_dist = torch.LongTensor([item[1] for item in data_storage_dist[batch_idx]])
                
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(full_data_batch_dist)
                outputs = outputs.squeeze(1) 
                loss = nn.CrossEntropyLoss()(outputs, full_label_batch_dist)   # Compute loss
                
                _, train_predicted = torch.max(outputs.data, 1)
                train_total += full_label_batch_dist.size(0)
                train_correct += (train_predicted == full_label_batch_dist).sum().item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Gossiping by considering a line topology
                right_neighbor_id = (agent_id + 1) % num_agents
                for param_name, param in model.named_parameters():
                    param.data = 0.5 * (param.data + models[right_neighbor_id].state_dict()[param_name])
                left_neighbor_id = (agent_id - 1) % num_agents
                for param_name, param in model.named_parameters():
                    param.data = 0.5 * (param.data + models[left_neighbor_id].state_dict()[param_name])
                
                average_train_accuracy = average_train_accuracy+ 100 * train_correct / train_total
                
                break  
      

        average_train_loss = loss
        average_train_accuracy = average_train_accuracy/num_agents

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
        
        # Applying proposed algorithm (Save the current model parameters for other system)
        if counter > 0:        # Save the current model parameters for the next iteration
            for current_param, cent_param in zip(global_model.parameters(), cent_model.parameters()):
                current_param.data = (1/(1/loss_cent+1/loss))*(1/loss * current_param.data + 1/loss_cent * cent_param.data)

        # Update the counter for each agent
        counter += 1
        if counter % dist_update_interval == 0:
            cent_comm = True
            counter = 0        
    
    ## Testing using MNIST dataset
    criterion = nn.CrossEntropyLoss()
    test_accuracy, test_loss = test(global_model, test_datasetLoader, criterion)
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)
           


# In[ ]:


# Plot the loss and accuracy curves for different num_agents
plt.figure(figsize=(12, 5))
plt.plot(test_accuracies, label=f'Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()

plt.figure(figsize=(12, 5))
plt.plot(test_losses, label=f'Average loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()






