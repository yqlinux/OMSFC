function [beta, history] = lassoRegression(X, y, lambda, max_iter, tol)
    % LASSO回归实现
    [n, p] = size(X);
    beta = zeros(p, 1);
    history = zeros(max_iter, 1);
    
    % 标准化
    X_std = (X - mean(X)) ./ std(X);
    y_std = (y - mean(y)) ./ std(y);
    
    for iter = 1:max_iter
        beta_old = beta;
        
        % 坐标下降
        for j = 1:p
            r = y_std - X_std * beta + X_std(:,j) * beta(j);
            beta(j) = softThreshold(X_std(:,j)' * r / n, lambda);
        end
        
        % 记录历史
        history(iter) = 0.5 * norm(y_std - X_std * beta)^2 + lambda * norm(beta, 1);
        
        % 检查收敛
        if norm(beta - beta_old) < tol
            break
        end
    end
    
    % 还原到原始尺度
    beta = beta ./ std(X)';
end


function z = softThreshold(x, lambda)
    % 软阈值函数
    z = sign(x) .* max(abs(x) - lambda, 0);
end