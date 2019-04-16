function [x,y,res] = feedforward(X,W1,W2,B1,B2)

x = [];
y = [];
res = [];

for i = 1 : size(X,2) 
    Y0 = X(:,i);
    
    U1 = W1 * Y0 + B1;
    Y1 = tanh(U1);

    U2 = W2 * Y1 + B2;
    Y2 = tanh(U2);

    x = [x, Y0(1,:)];
    %y = [y, Y0(2,:)];
    res = [res, Y2];
end
end