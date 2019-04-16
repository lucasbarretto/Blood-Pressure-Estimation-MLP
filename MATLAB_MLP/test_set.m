function MSE = test_set(X,Z,W1,W2,B1,B2)

J = [];

for index = 1:size(X,2)
    
    % feedforward
    Y0 = X(:,index);
    D = Z(index);

    U1 = W1 * Y0 + B1;
    Y1 = tanh(U1);

    U2 = W2 * Y1 + B2;
    Y2 = tanh(U2);
    
    % calculates the quadratic error
    J = [J, (D - Y2)^2];
    
end

MSE = sum(J)/size(J,2);

end