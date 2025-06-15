function y = shuxingyuejian(A, D)

%%
[m, n] = size(A);
    
try
    A_std = (A - mean(A)) ./ std(A);
    covariance = A_std' * A_std / m;
    alpha = 0.15;
    max_iter = 150;
    tol = 1e-7;
    [loadings, ~] = sparsePCA(covariance, min(n, round(m*0.5)), alpha, max_iter, tol);
    feature_scores = sum(abs(loadings), 2);
    [sorted_scores, idx] = sort(feature_scores, 'descend');
    cutoff = sorted_scores(max(1, round(n * 0.4)));
    important_features = feature_scores >= cutoff;
catch
    warning('Sparse PCA计算失败，使用所有特征继续。');
    important_features = ones(n, 1);
end
    
try
    lambda = 0.02;
    beta_matrix = zeros(n, size(D, 2));
    
    for i = 1:size(D, 2)
        [beta, ~] = lassoRegression(A, D(:,i), lambda, 200, 1e-7);
        beta_matrix(:,i) = beta;
    end
    
    lasso_scores = sum(abs(beta_matrix), 2);
    [sorted_scores, idx] = sort(lasso_scores, 'descend');
    cutoff = sorted_scores(max(1, round(n * 0.4)));
    lasso_features = lasso_scores >= cutoff;
catch
    warning('LASSO计算失败，使用Sparse PCA结果继续。');
    lasso_features = important_features;
end
    
combined_importance = important_features | lasso_features;
    
Aold = A;
Core_A = zeros(n, n);
old_Core_XA = [];
allA_Core_A = zeros(n, n);
    
for i = 1:n
    if ~combined_importance(i)
        continue;
    end
        
    A_new = Aold;
    A_new(:,i) = [];
    if 0 < Sig(Aold, A_new, D)
        Core_A(i,i) = 1;
        old_Core_XA = [old_Core_XA Aold(:,i)];
    else
        allA_Core_A(i,i) = 1;
    end
end
    
oldCore_A = Core_A;
    
for j = 1:n
    ii = [];
    temsumsigout = [];
        
    for i = 1:n
        if allA_Core_A(i,i) == 1 && combined_importance(i)
            Core_A(i,i) = 1;
            ii = [ii i];
            temsumsigout = [temsumsigout sigout(oldCore_A, Core_A)];
        end
    end
        
    if isempty(temsumsigout)
        continue
    end
        
    sumsigout = [ii; temsumsigout];
    [~, I] = max(sumsigout(2,:));
        
    if size(sumsigout,2) < 1
        continue
    end
        
    oldCore_A(sumsigout(1,I), sumsigout(1,I)) = 1;
    allA_Core_A(sumsigout(1,I), sumsigout(1,I)) = 0;
        
    old_Core_XA = [old_Core_XA A(:,sumsigout(1,I))];
    try
        if tiaojian(old_Core_XA, D) == tiaojian(A, D)
            break
        end
    catch
        break
    end
end
    
for i = 1:n
    if oldCore_A(i,i) == 1 && Sigin(A, oldCore_A(i,i)) == 0
        oldCore_A(i,i) = 0;
    end
end
    
y = oldCore_A;
end