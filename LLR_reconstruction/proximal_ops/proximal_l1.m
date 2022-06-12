function x = proximal_l1(a, lambda)
% minimizes for x
%   lambda * |x|_1 + 1/2 * (x - a)^2
    x =  sign(a) .* max(abs(a) - lambda, 0);
end