function [Opt, timing] = OptimalScaleFC1(tmpdata, data)

%%
timing = struct();
tic;
dataLength = length(tmpdata{1, 1});
Datas = cell(1, dataLength);
DecisonGC = unique(Granular_concept(D2C(data)), 'rows', 'stable');
timing.initialization = toc;

tic;
parfor k = 1:dataLength
    initialContext = Breakpoint2Context(tmpdata{1,1}{k}, data, k);
    CV0 = SubFun1(initialContext, data, DecisonGC);
    OptGra = tmpdata{1,1}{k};
    
    up = tmpdata{1,size(tmpdata,2)}{k};
    for layer = (size(tmpdata,2)):-1:2
        down = tmpdata{1,layer-1}{k};
        Diff1 = setdiff(down, up);
        
        if isempty(Diff1)
            continue;
        end
        
        allPoints = arrayfun(@(x) unique([up Diff1(1:x)]), 1:length(Diff1), 'UniformOutput', false);
        allContexts = cellfun(@(x) Breakpoint2Context(x, data, k), allPoints, 'UniformOutput', false);
        allMRIs = cellfun(@(x) SubFun1(x, data, DecisonGC), allContexts);
        
        idx = find(allMRIs >= CV0, 1);
        if ~isempty(idx)
            OptGra = allPoints{idx};
            break;
        end
    end
    
    Datas{k} = Breakpoint2Context(OptGra, data, k);
end
timing.main_loop = toc;

tic;
Datas = cell2mat(Datas);

y = shuxingyuejian(Datas, D2C(data));
diag_elements = diag(y);
Opt = Datas(:, diag_elements ~= 0);

if isempty(Opt) || size(Opt, 2) == 0
    Opt = Datas;
end
timing.post_processing = toc;

timing.total_time = timing.initialization + timing.main_loop + timing.post_processing;

fprintf('\n运行时间统计 (秒):\n');
fprintf('初始化阶段: %.4f\n', timing.initialization);
fprintf('主循环处理: %.4f\n', timing.main_loop);
fprintf('后处理阶段: %.4f\n', timing.post_processing);
fprintf('总运行时间: %.4f\n', timing.total_time);
end


%% SubFunctions
function y = SubFun1(Context, data, precomputedDecisonGC)
ConditionGC = unique(Granular_concept(Context), 'rows', 'stable');
G = size(data, 1);
epsilon = 1e-10;

DecisionCounts = sum(precomputedDecisonGC(:, 1:G), 2);
Pf = DecisionCounts / G;
EntS = -sum(Pf .* log2(max(Pf, epsilon))) / size(precomputedDecisonGC, 1);

ObjCC_matrix = ConditionGC(:, 1:G);
jointCounts = precomputedDecisonGC(:, 1:G) * ObjCC_matrix' / G;
validCounts = jointCounts > epsilon;
CIE = sum(-jointCounts .* log2(max(jointCounts, epsilon)) .* validCounts, 1);
InfoGain = EntS - CIE;

totalInfoGain = sum(InfoGain);
if totalInfoGain > epsilon
    InfoGainWeights = InfoGain / totalInfoGain;
else
    InfoGainWeights = ones(1, size(ConditionGC, 1)) / size(ConditionGC, 1);
end

y = mean(InfoGain .* InfoGainWeights);
end

function y = Granular_concept(X)
[m0, n0] = size(X);
concept = zeros(m0, m0 + n0);

X_logical = logical(X);
[rows, cols] = find(X_logical);
unique_rows = unique(rows);

X_sparse = sparse(X_logical);
for i = unique_rows'
    idx = cols(rows == i);
    concept(i, idx + m0) = 1;
    concept(i, 1:m0) = all(X_sparse(:,idx) == X_sparse(i,idx), 2)';
end

y = concept;
end

function Context = Breakpoint2Context(breakpoint, data, k)
[rows, ~] = size(data);
num_breakpoints = size(breakpoint, 2) - 1;
data_k = data(:, k);

lower_bounds = breakpoint(1, 1:end-1);
upper_bounds = breakpoint(1, 2:end);

Context = false(rows, num_breakpoints);
Context(:,1:num_breakpoints-1) = data_k >= lower_bounds(1:end-1) & data_k < upper_bounds(1:end-1);
Context(:,num_breakpoints) = data_k >= lower_bounds(end) & data_k <= upper_bounds(end);

Context = Context(:, any(Context, 1));
end

function D = D2C(data)
[classes, ~, idx] = unique(data(:, end));
numClasses = length(classes);
numSamples = size(data, 1);

D = sparse(1:numSamples, idx, true, numSamples, numClasses);
D = full(D);

noClass = ~any(D, 2);
D(noClass, end) = true;
end

function y = shuxingyuejian(A, D)
[m, n] = size(A);

try
    A_std = (A - mean(A)) ./ std(A);
    covariance = A_std' * A_std / m;
    alpha = 0.15;
    max_iter = 150;
    tol = 1e-7;
    [loadings, ~] = sparsePCA(covariance, min(n, round(m*0.5)), alpha, max_iter, tol);
    feature_scores = sum(abs(loadings), 2);
    [sorted_scores, ~] = sort(feature_scores, 'descend');
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
    [sorted_scores, ~] = sort(lasso_scores, 'descend');
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

function [loadings, eigenvalues] = sparsePCA(X, k, alpha, max_iter, tol)
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

function [beta, history] = lassoRegression(X, y, lambda, max_iter, tol)
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

function z = softThreshold(x, lambda)
z = sign(x) .* max(abs(x) - lambda, 0);
end