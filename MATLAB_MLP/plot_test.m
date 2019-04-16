[x_res,y_res,z_res] = feedforward(X_train_normal, W1, W2, B1, B2);
hold on
%scatter3(x_res, y_res, z_res,'.')
scatter(x_res, z_res,'.')
%scatter3(X_train_normal(1,:),X_train_normal(2,:),Z_train_normal,'.')
scatter(X_train_normal(1,:),Z_train_normal,'.')
hold off