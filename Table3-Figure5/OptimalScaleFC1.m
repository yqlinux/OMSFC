function Opt = OptimalScaleFC1(tmpdata, data)

%%
Datas = [];
parfor k = 1:length(tmpdata{1, 1})
    [CV0] = SubFun1(Breakpoint2Context(tmpdata{1,1}{k}, data, k), data);
    OptGra = tmpdata{1,1}{k};
    
    up=tmpdata{1,size(tmpdata,2)}{k};
    for layer=(size(tmpdata,2)):-1:2
        down=tmpdata{1,layer-1}{k};

        Diff1=setdiff(down, up);
        for num=1:length(Diff1)
            TmpBreakPoint=unique([up Diff1(1:num)]);

            [MRI] = SubFun1(Breakpoint2Context(TmpBreakPoint, data, k), data);

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

if isempty(Opt) || size(Opt, 2) == 0
    Opt = Datas;
end

end


%% SubFunctions
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

function y = SubFun1(Context, data)
ConditionGC = unique(Granular_concept(Context), 'rows', 'stable');
DecisonGC = unique(Granular_concept(D2C(data)), 'rows', 'stable');

G = size(data, 1);
epsilon = 1e-10;
InfoGain = zeros(1, size(ConditionGC, 1));

for i = 1:size(ConditionGC, 1)
    ObjCC = ConditionGC(i, 1:G);
    CIE = 0;
    EntS = 0;
    for j = 1:size(DecisonGC, 1)
        ObjDC = DecisonGC(j, 1:G);
        Pf = length(find(ObjDC == 1)) / G;
        EntS = EntS + -Pf * log2(max(Pf, epsilon));

        Pcf = length(find((ObjDC & ObjCC) == 1)) / G;
        CIE = CIE + -Pcf * log2(max(Pcf, epsilon));
    end

    EntS = EntS / size(DecisonGC, 1);
    InfoGain(i) = EntS - CIE;
end

totalInfoGain = sum(InfoGain);
if totalInfoGain > epsilon
    InfoGainWeights = InfoGain / totalInfoGain;
else
    InfoGainWeights = ones(1, size(ConditionGC, 1)) / size(ConditionGC, 1);
end
   
WeightedGain = InfoGain .* InfoGainWeights;
y = mean(WeightedGain);
end

function y=Granular_concept(X)
[m0,n0]=size(X);
concept=[];
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
[rows, ~] = size(data);
num_breakpoints = size(breakpoint, 2) - 1;

Context = zeros(rows, num_breakpoints);

for j = 1:num_breakpoints
    lower_bound = breakpoint(1, j);
    upper_bound = breakpoint(1, j + 1);

    tmp = (data(:, k) >= lower_bound) & (data(:, k) < upper_bound);
    if num_breakpoints==j
        tmp=tmp | (data(:, k) == upper_bound);
    end

    Context(:, j) = tmp;
end
Context = Context(:, any(Context ~= 0));
end

function D = D2C(data)
classes = unique(data(:, end));
D = zeros(size(data, 1), length(classes));

for j = 1:length(classes)
    classIndices = data(:, end) == classes(j);
    D(classIndices, j) = 1;
end

D(~any(D, 2), end) = 1;
end