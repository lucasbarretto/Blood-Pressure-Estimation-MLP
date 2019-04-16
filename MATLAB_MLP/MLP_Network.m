clearvars -except Part_1
%load Part_1;

% Setting values for the hyperparameters
i = 1; % number of inputs
n = 50; % number of neurons in the hidden layer
epsilon = 0.00005; % minimum range for quadratic error
alpha = 0.001; % learning rate
max_epochs = 1000000; % maximum number of epochs

% Initializes the dataset input and shuffles it
%raw_input = shuffle(Part_1{1,1});
raw_input = -1+2*rand(1,1000);
z = (raw_input(1,:)).^2;
input_size = size(raw_input, 2);

% Initializes the training set's input and label matrices - 70%
%X_train = raw_input(1:2:3, 1 : 0.7*input_size);
%Z_train = raw_input(2, 1 : 0.7*input_size);

X_train = raw_input(:,1:0.7*input_size);
Z_train = z(:,1:0.7*input_size);

% Initializes the test set's input and label matrices - 30%
%X_test = raw_input(1:2:3, (0.7*input_size + 1) : input_size);
%Z_test = raw_input(2, (0.7*input_size + 1) : input_size);
X_test = raw_input(:, (0.7*input_size + 1) : input_size);
Z_test = raw_input(:, (0.7*input_size + 1) : input_size);

% Initializes the weights and biases matrices w/ random values between [-0.1, 0.1]
W1 = -0.1 + 0.2 * rand(n,i);
W2 = -0.1 + 0.2 * rand(1, n);

B1 = -0.1 + 0.2 * rand(n,1);
B2 = -0.1 + 0.2 * rand(1,1);

% Normalizes the input and result matrices for both sets
X_train_normal = -1 + 2*normalize(X_train, 'range');
Z_train_normal = -1 + 2*normalize(Z_train, 'range');
X_test_normal = -1 + 2*normalize(X_test, 'range');
Z_test_normal = -1 + 2*normalize(Z_test, 'range');

% Training the network and testing the results
J_train = [];
J_test = [];

for epoch = 1:max_epochs
    [j_train,W1,W2,B1,B2] = backpropagation_step(X_train_normal,Z_train_normal,W1,W2,B1,B2,epsilon,alpha);
    J_train = [J_train, j_train];
    
    j_test = test_set(X_test_normal,Z_test_normal,W1,W2,B1,B2);
    J_test = [J_test, j_test];
    
    if mod(epoch, 5) == 0
        fprintf('epoch %d - train error: %4.6f | test error: %4.4f\n',epoch,j_train, j_test)
    end
    
    if j_train < epsilon
        break
    end
end

plot(J_train, 'b')
hold on
plot(J_test, 'r')
title(['MLP Network with ' num2str(n) ' neuron(s) in the hidden layer | alpha = ' num2str(alpha)])
hold off