function Opt = OptimalScaleFC1(tmpdata, data)

%% MainCodes
Datas = [];
parfor k = 1:length(tmpdata{1, 1}) % 遍历每棵粒度树
    [CV0] = SubFun1(Breakpoint2Context(tmpdata{1,1}{k}, data, k), data);
    OptGra = tmpdata{1,1}{k};
    
    up=tmpdata{1,size(tmpdata,2)}{k};% 上层粒度树
    for layer=(size(tmpdata,2)):-1:2% 从粒度树第二层开始遍历剪枝，使用下层节点替换当前节点（不同粒度层替换只考虑两层之间）
        down=tmpdata{1,layer-1}{k};% 下层粒度树

        Diff1=setdiff(down, up);% 细和粗之间的差 
        for num=1:length(Diff1)% 按顺序依次增加个数
            TmpBreakPoint=unique([up Diff1(1:num)]);

            [MRI] = SubFun1(Breakpoint2Context(TmpBreakPoint, data, k), data); % 计算新组合的准则值

            if MRI >= CV0
                OptGra = TmpBreakPoint;
                break
            end
        end
    end

    Context = Breakpoint2Context(OptGra, data,k);
    S = 0;
    for ii=1:size(Context,2)
        S = S + Context(:,ii)*ii;
    end
    Datas = [Datas Context];
end

y=shuxingyuejian(Datas,D2C(data));
diag_elements = diag(y);
Opt = Datas(:,diag_elements ~= 0);

if isempty(Opt) || size(Opt, 2) == 0  % 确保不仅判断 [0×0] 的空情况
    Opt = Datas;
end
end


%%
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


function y = SubFun1(Context, data)
    ConditionGC = unique(Granular_concept(Context), 'rows', 'stable');
    DecisonGC = unique(Granular_concept(D2C(data)), 'rows', 'stable');

    G = size(data, 1);  % 样本数量
    epsilon = 1e-10;    % 防止对数运算出现NaN
    InfoGain = zeros(1, size(ConditionGC, 1));    % 初始化信息增益数组

    % 计算每个粒度概念的信息增益
    for i = 1:size(ConditionGC, 1)
        ObjCC = ConditionGC(i, 1:G);  % 条件粒概念外延
        % PC = length(find(ObjCC == 1)) / G;  % 条件粒度概念的概率 p(f)

        CIE = 0;  % 条件互信息
        EntS = 0; % 信息熵
        for j = 1:size(DecisonGC, 1)
            ObjDC = DecisonGC(j, 1:G);  % 决策粒概念外延
            Pf = length(find(ObjDC == 1)) / G;  % 决策粒度概念的概率 p(f)
            EntS = EntS + -Pf * log2(max(Pf, epsilon));% 计算决策的熵

            % 计算条件互信息
            Pcf = length(find((ObjDC & ObjCC) == 1)) / G;  % 条件互信息
            CIE = CIE + -Pcf * log2(max(Pcf, epsilon));
        end

        EntS = EntS / size(DecisonGC, 1);
        InfoGain(i) = EntS - CIE;% 计算信息增益
    end

    % 正常化信息增益，使其和为1，作为加权因子
    totalInfoGain = sum(InfoGain);
    if totalInfoGain > epsilon
        InfoGainWeights = InfoGain / totalInfoGain;  % 每个粒度概念的权重
    else
        InfoGainWeights = ones(1, size(ConditionGC, 1)) / size(ConditionGC, 1);  % 默认均匀分配
    end
   
    WeightedGain = InfoGain .* InfoGainWeights; % 对增益进行加权
    y = mean(WeightedGain);% 最终结果：计算加权后的增益的平均值
end


function y=Granular_concept(X)%对象粒概念

[m0,n0]=size(X);%m0:对象个数，n0：属性个数
concept=[];
%m0:对象数；n0：属性数
for i=1:m0
    x=[zeros(1,m0+n0)];
    A=[];
    for j=1:m0
    A=find(X(i,:)==1);
    x(A'+m0)=1;
    if all(X(j,A)==X(i,A))
       x(j)=1;
    end
   end
concept=[concept;x];
end

y=concept;
end


function Context = Breakpoint2Context(breakpoint, data, k)
% Breakpoint2Context: 计算与断点对应属性k的形式背景
% breakpoint: 数据集的断点
% data: 原数据
% k: 第k个属性

% 获取数据的行数和断点的数量
[rows, ~] = size(data);
num_breakpoints = size(breakpoint, 2) - 1;

Context = zeros(rows, num_breakpoints); % 为形式背景预分配空间

for j = 1:num_breakpoints
    % 获取当前断点的范围
    lower_bound = breakpoint(1, j);
    upper_bound = breakpoint(1, j + 1);

    tmp = (data(:, k) >= lower_bound) & (data(:, k) < upper_bound);
    % 计算当前属性的形式背景
    if num_breakpoints==j
        tmp=tmp | (data(:, k) == upper_bound);
    end

    % 将计算得到的形式背景添加到结果矩阵中
    Context(:, j) = tmp;
end
Context = Context(:, any(Context ~= 0));% 删除全为0的列
end


function D = D2C(data)
% D2C: 分类标签转换为形式背景
% data: 原数据

classes = unique(data(:, end));
D = zeros(size(data, 1), length(classes));

for j = 1:length(classes)
    classIndices = data(:, end) == classes(j);
    D(classIndices, j) = 1;
end

D(~any(D, 2), end) = 1; % 处理类别为0的情况
end

