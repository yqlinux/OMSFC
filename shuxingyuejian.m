function y = shuxingyuejian(A, D)
    % 属性约简函数
    % A: 条件属性矩阵
    % D: 决策属性矩阵
    
    [m, n] = size(A);
    
    % 第一步：使用Sparse PCA进行特征选择
    try
        % 简单的数据标准化
        A_std = (A - mean(A)) ./ std(A);
        
        % 计算协方差矩阵
        covariance = A_std' * A_std / m;
        
        % Sparse PCA参数
        alpha = 0.15;  % 增加稀疏性
        max_iter = 150;
        tol = 1e-7;
        
        % 执行Sparse PCA
        [loadings, ~] = sparsePCA(covariance, min(n, round(m*0.5)), alpha, max_iter, tol);
        
        % 计算特征重要性
        feature_scores = sum(abs(loadings), 2);
        
        % 使用固定比例选择特征
        [sorted_scores, idx] = sort(feature_scores, 'descend');
        cutoff = sorted_scores(max(1, round(n * 0.4))); % 保留前40%的特征
        important_features = feature_scores >= cutoff;
    catch
        warning('Sparse PCA计算失败，使用所有特征继续。');
        important_features = ones(n, 1);
    end
    
    % 第二步：使用LASSO进行特征选择
    try
        % 固定的LASSO参数
        lambda = 0.02;
        beta_matrix = zeros(n, size(D, 2));
        
        for i = 1:size(D, 2)
            [beta, ~] = lassoRegression(A, D(:,i), lambda, 200, 1e-7);
            beta_matrix(:,i) = beta;
        end
        
        % 基于LASSO系数选择特征
        lasso_scores = sum(abs(beta_matrix), 2);
        [sorted_scores, idx] = sort(lasso_scores, 'descend');
        cutoff = sorted_scores(max(1, round(n * 0.4))); % 保留前40%的特征
        lasso_features = lasso_scores >= cutoff;
    catch
        warning('LASSO计算失败，使用Sparse PCA结果继续。');
        lasso_features = important_features;
    end
    
    % 合并特征选择结果（使用或运算）
    combined_importance = important_features | lasso_features;
    
    % 传统属性约简过程
    Aold = A;
    Core_A = zeros(n, n);
    old_Core_XA = [];
    allA_Core_A = zeros(n, n);
    
    % 初始化核心属性
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
    
    % 迭代优化过程
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
    
    % 最终优化：去除冗余属性
    for i = 1:n
        if oldCore_A(i,i) == 1 && Sigin(A, oldCore_A(i,i)) == 0
            oldCore_A(i,i) = 0;
        end
    end
    
    y = oldCore_A;
end
