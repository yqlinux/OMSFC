function [beta, history] = lassoRegression(X, y, lambda, max_iter, tol)

%%
[n, p] = size(X);
beta = zeros(p, 1);
history = zeros(max_iter, 1);

X_std = (X - mean(X)) ./ std(X);
y_std = (y - mean(y)) ./ std(y);

for iter = 1:max_iter
    beta_old = beta;
    
    for j = 1:p
        r = y_std - X_std * beta + X_std(:,j) * beta(j);
        beta(j) = softThreshold(X_std(:,j)' * r / n, lambda);
    end
    
    history(iter) = 0.5 * norm(y_std - X_std * beta)^2 + lambda * norm(beta, 1);
    
    if norm(beta - beta_old) < tol
        break
    end
end

beta = beta ./ std(X)';

end


%% SubFunctions
function z = softThreshold(x, lambda)
z = sign(x) .* max(abs(x) - lambda, 0);
end