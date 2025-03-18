function [loadings, eigenvalues] = sparsePCA(X, k, alpha, max_iter, tol)
    % 稀疏主成分分析实现
    [n, p] = size(X);
    loadings = randn(p, k);
    loadings = loadings ./ sqrt(sum(loadings.^2));
    
    for iter = 1:max_iter
        % 更新scores
        scores = X * loadings;
        
        % 更新loadings
        old_loadings = loadings;
        for j = 1:k
            % 软阈值操作
            r = X' * scores(:,j);
            loadings(:,j) = sign(r) .* max(abs(r) - alpha, 0);
            loadings(:,j) = loadings(:,j) / norm(loadings(:,j));
        end
        
        % 检查收敛
        if norm(loadings - old_loadings, 'fro') < tol
            break
        end
    end
    
    % 计算特征值
    eigenvalues = diag(scores' * scores / n);
end
