function [MSE,W1,W2,B1,B2] = backpropagation_step(X,Z,W1,W2,B1,B2,epsilon,alpha)

W1_update = 0;
W2_update = 0;
B1_update = 0;
B2_update = 0;
J = [];

for index = 1:size(X,2)
    % feedforward
    Y0 = X(:,index);
    D = Z(index);

    U1 = W1 * Y0 + B1;
    Y1 = tanh(U1);

    U2 = W2 * Y1 + B2;
    Y2 = tanh(U2);

    % backpropagation
    F_diff_2 = get_F_diff(U2);
    delta_2 = -2*F_diff_2*(D - Y2);

    F_diff_1 = get_F_diff(U1);
    delta_1 = F_diff_1*W2'*delta_2;

    W2_update = W2_update + delta_2*Y1';
    W1_update = W1_update + delta_1*Y0';
    
    B2_update = B2_update + delta_2;
    B1_update = B1_update + delta_1;
    
    % calculates the quadratic error
    J = [J, (D - Y2)^2];
end

% updates weights and biases
W2 = W2 - alpha*W2_update;
B2 = B2 - alpha*B2_update;

W1 = W1 - alpha*W1_update;
B1 = B1 - alpha*B1_update;

MSE = sum(J)/size(J,2);

end