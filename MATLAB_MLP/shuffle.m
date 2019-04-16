function X_shuffled = shuffle(X)

[r c] = size(X);
shuffled_columns = randperm(c);
X_shuffled = X(:, shuffled_columns);
end