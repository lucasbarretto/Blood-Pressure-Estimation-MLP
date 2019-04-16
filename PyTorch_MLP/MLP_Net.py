import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# loads the data set
data = np.loadtxt(open("dataSet.csv", "rb"), delimiter=",")

# defines the training dataset
X_train = data[0:2,0:10000]
Z_train = data[2,0:10000]

# defines the testing dataset
X_test = data[0:2,10001:15000]
Z_test = data[2,10001:15000]

# normalizing function
def normalize(A):
    return -1 + 2*(A-min(A))/(max(A)-min(A))

# normalizes the trainin dataset
X_train_normal = np.array([normalize(X_train[0,:]), normalize(X_train[1,:])])
X_train_normal = X_train_normal.transpose()
x_train = torch.from_numpy(X_train_normal)

Z_train_normal = np.array(normalize(Z_train))
Z_train_normal = Z_train_normal.transpose()
z_train = torch.from_numpy(Z_train_normal)

# normalizes the testing dataset
X_test_normal = np.array([normalize(X_test[0,:]), normalize(X_test[1,:])])
X_test_normal = X_test_normal.transpose()
x_test = torch.from_numpy(X_test_normal)

Z_test_normal = np.array(normalize(Z_test))
Z_test_normal = Z_test_normal.transpose()
z_test = torch.from_numpy(Z_test_normal)

# defines the neural network
class Net(nn.Module):
    def __init__(self,i,n):
        super(Net, self).__init__()
        self.l1 = nn.Linear(i,n,bias=True)
        self.l2 = nn.Linear(n,1, bias=True)
        
    def forward(self, X):
        y1 = torch.tanh(self.l1(X))
        y2 = torch.tanh(self.l2(y1))
        return y2

# defines the hyperparameters
i = 2
n = 10
eps = 10**-5
max_epochs = 10**4

# constructs the model
model = Net(i,n)

# defines criterion
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# trains the network
loss_train_data = []
loss_test_data = []

for epoch in range(max_epochs):
            
    # forward propagation
    train_pred = model(x_train.float())
    test_pred = model(x_test.float())
        
    # computes the loss
    train_loss = criterion(train_pred, z_train)
    loss_train_data.append(train_loss)
    
    test_loss = criterion(train_pred, z_test)
    loss_test_data.append(test_loss)
        
    # zeroes the parameter gradients
    optimizer.zero_grad()
        
    # backpropagation and updates the weights
    train_loss.backward()
    optimizer.step()
    
    print('epoch ' + str(epoch) + ' - MSE_train: %.5f | MSE_test: %.5f' %  (train_loss.item(), test_loss.item()))
    
    if train_loss < eps:
        break
    
    break