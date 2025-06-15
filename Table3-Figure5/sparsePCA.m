function [loadings, eigenvalues] = sparsePCA(X, k, alpha, max_iter, tol)

%%
[n, p] = size(X);
loadings = randn(p, k);
loadings = loadings ./ sqrt(sum(loadings.^2));

for iter = 1:max_iter
    scores = X * loadings;
    
    old_loadings = loadings;
    for j = 1:k
        r = X' * scores(:,j);
        loadings(:,j) = sign(r) .* max(abs(r) - alpha, 0);
        loadings(:,j) = loadings(:,j) / norm(loadings(:,j));
    end
    
    if norm(loadings - old_loadings, 'fro') < tol
        break
    end
end

eigenvalues = diag(scores' * scores / n);

end