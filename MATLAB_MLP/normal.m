function X_normal = normal(X)
X_normal = [];

for a = 1:size(X,1)
    m = mean(X(a,:));
    sd = std(X(a,:));
    X_normal = [X_normal;(X(a,:) - m)/sd];
end
end