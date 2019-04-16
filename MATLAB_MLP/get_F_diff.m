function F_diff = get_F_diff(U)
F_diff = eye(size(U,1));
b = 1;

for a = 1:size(F_diff,1)
    F_diff(a,a) = 1-tanh(U(b))^2;
    b = b + 1;
end
end