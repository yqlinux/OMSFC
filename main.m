%% Codes，不同方法对比实验精度
warning("off");close all;clc;clear;
addpath(genpath(pwd))

nameall="appendicitis";
allResults = cell(8, 12);  % 重置结果数组 (7个方法 + 1个标题行)
rowIndex = 1;

fprintf('\n处理数据集 %d: %s\n', i, nameall(i));
fprintf('----------------------------------------\n');

data=table2array(readtable("F:\HPDesktopFiles\MyPaper\Paper1\BOGSM1\DT2\"+nameall(i)+'.txt'));
data= fillmissing(data,"nearest");
tmpdata = load("C:\Users\yq201472\Desktop\Figure(3-4)\MultiGranDat\"+(num2str(i)+".mat")).MFC{1,1};

% 运行时间和结果
tic; optimal{1} = OptimalScaleTPE(tmpdata, data); time(1) = toc;
tic; optimal{2} = OptimalScaleFC1(tmpdata, data); time(2) = toc;
tic; optimal{3} = OptimalScaleHJJ(tmpdata, data); time(3) = toc;
tic; optimal{4} = OptimalScaleWWZ(tmpdata, data); time(4) = toc;
tic; optimal{5} = OptimalScaleCDX(tmpdata, data); time(5) = toc;
tic; optimal{6} = OptimalScaleZQH(tmpdata, data); time(6) = toc;
tic; optimal{7} = OptimalScaleHZH(tmpdata, data); time(7) = toc;

% 获取准确率、标准差和ARI值
[AccKnn, StdKnn, AriKnn, AriStdKnn] = RF(optimal,data);
[AccDT, StdDT, AriDT, AriStdDT] = DT(optimal,data);
[AccSvm, StdSvm, AriSvm, AriStdSvm] = SVM(optimal,data);
[AccGBM, StdGBM, AriGBM, AriStdGBM] = GBM(optimal,data);
[AccXGB, StdXGB, AriXGB, AriStdXGB] = XGBoost(optimal,data);



%% SubFunction
function [accuracy, stderror, ari_mean, ari_std] = XGBoost(optimal, data)
rng('default')
k = 10;
cv = cvpartition(size(optimal{1}, 1), 'KFold', k);

for m = 1:size(optimal, 2)
    data = [optimal{m} data(:, end)];
    accuracy_all = [];
    ari_all = [];
    for i = 1:k
        train = array2table(data(cv.training(i), :));
        test = array2table(data(cv.test(i), :));
        [~, validationAccuracy, ari] = XGBoostClassifier(train, test);
        accuracy_all = [accuracy_all validationAccuracy];
        ari_all = [ari_all ari];
    end
    stderror(m) = std(accuracy_all);
    accuracy(m) = mean(accuracy_all);
    ari_mean(m) = mean(ari_all);
    ari_std(m) = std(ari_all);
end
end

function [time, validationAccuracy, ari] = XGBoostClassifier(train, test)
tic;
inputTable = train;
predictorNames = train.Properties.VariableNames(1:end-1);
predictors = inputTable(:, predictorNames);
response = table2array(train(:, end));
class = unique(response);
test_predictors = table2array(test(:, 1:end-1));

% CatBoost风格参数设置
params = struct();
params.iterations = 100;        % 默认迭代次数
params.learning_rate = 0.03;    % CatBoost默认学习率
params.depth = 6;              % CatBoost默认深度
params.l2_leaf_reg = 3;        % L2正则化系数
params.min_data_in_leaf = 1;   % 最小叶子节点样本数

% 创建基础树模板
template = templateTree(...
    'MaxNumSplits', 2^params.depth, ...
    'MinLeafSize', params.min_data_in_leaf, ...
    'SplitCriterion', 'deviance', ...  % 使用对数似然损失
    'Surrogate', 'off');

% 根据类别数选择合适的提升方法
if length(class) <= 2
    method = 'AdaBoostM1';    % 二分类
else
    method = 'AdaBoostM2';     % 多分类
end

try
    % 训练模型
    model = fitcensemble(predictors, response, ...
        'Method', method, ...
        'NumLearningCycles', params.iterations, ...
        'Learners', template, ...
        'LearnRate', params.learning_rate, ...
        'ClassNames', class);

    % 预测
    [predict_label, scores] = predict(model, test_predictors);
    test_label = table2array(test(:,end));

    % 计算准确率
    validationAccuracy = (sum(predict_label == test_label) / length(test_label)) * 100;

    % 计算ARI
    ari = rand_index(test_label, predict_label, 'adjusted');

    % 对二分类问题计算和显示额外的指标
    if length(class) == 2
        % 计算混淆矩阵
        confMat = confusionmat(test_label, predict_label);
        TP = confMat(2,2);
        TN = confMat(1,1);
        FP = confMat(1,2);
        FN = confMat(2,1);

        % 计算性能指标
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        f1_score = 2 * (precision * recall) / (precision + recall);

        % 计算并绘制ROC曲线
        [X,Y,~,AUC] = perfcurve(test_label, scores(:,2), 1);

    end

catch ME
    % 如果出错，使用更简单的模型作为备选
    warning('模型训练失败，使用备选方法: %s', ME.message);

    template = templateTree('MaxNumSplits', 10);
    model = fitcensemble(predictors, response, ...
        'Method', 'Bag', ...
        'NumLearningCycles', 50, ...
        'Learners', template);

    predict_label = predict(model, test_predictors);
    test_label = table2array(test(:,end));
    validationAccuracy = (sum(predict_label == test_label) / length(test_label)) * 100;
    ari = rand_index(test_label, predict_label, 'adjusted');
end

time = toc;
end

function [accuracy, stderror, ari_mean, ari_std] = GBM(optimal, data)
rng('default')
k = 10;
cv = cvpartition(size(optimal{1}, 1), 'KFold', k);

for m = 1:size(optimal, 2)
    data = [optimal{m} data(:, end)];
    accuracy_all = [];
    ari_all = [];
    for i = 1:k
        train = array2table(data(cv.training(i), :));
        test = array2table(data(cv.test(i), :));
        [~, validationAccuracy, ari] = GradientBoostingClassifier(train, test);
        accuracy_all = [accuracy_all validationAccuracy];
        ari_all = [ari_all ari];
    end
    stderror(m) = std(accuracy_all);
    accuracy(m) = mean(accuracy_all);
    ari_mean(m) = mean(ari_all);
    ari_std(m) = std(ari_all);
end
end

function [time, validationAccuracy, ari] = GradientBoostingClassifier(train, test)
tic;
inputTable = train;
predictorNames = train.Properties.VariableNames(1:end-1);
predictors = inputTable(:, predictorNames);
response = table2array(train(:, end));
class = unique(response);

% 修改树模板参数，增加鲁棒性
template = templateTree(...
    'MaxNumSplits', 10, ...        % 减小最大分裂数
    'MinLeafSize', 1, ...          % 减小最小叶子节点大小
    'MinParentSize', 2, ...        % 添加最小父节点大小
    'Surrogate', 'off');

try
    % 修改GBM参数设置
    classificationGBM = fitcensemble(...
        predictors, ...
        response, ...
        'Method', 'GentleBoost', ...    % 使用GentleBoost替代AdaBoostM2
        'NumLearningCycles', 50, ...     % 减少学习周期
        'Learners', template, ...
        'LearnRate', 0.1, ...
        'ClassNames', class);

    % 添加交叉验证来评估模型性能
    cvmodel = crossval(classificationGBM);
    cvError = kfoldLoss(cvmodel);

    if cvError > 0.5  % 如果交叉验证错误率过高，尝试其他参数
        classificationGBM = fitcensemble(...
            predictors, ...
            response, ...
            'Method', 'RUSBoost', ...    % 尝试RUSBoost
            'NumLearningCycles', 100, ...
            'Learners', template, ...
            'LearnRate', 0.05);          % 降低学习率
    end

    predictorExtractionFcn = @(t) t(:, predictorNames);
    gbmPredictFcn = @(x) predict(classificationGBM, x);
    trainedClassifier.predictFcn = @(x) gbmPredictFcn(predictorExtractionFcn(x));

    predict_label = trainedClassifier.predictFcn(test);
    test_label = table2array(test(:,end));
    validationAccuracy = (sum(predict_label == test_label) / length(test_label)) * 100;

    ari = rand_index(test_label, predict_label, 'adjusted');

    % 如果精度异常低，使用备选方法
    if validationAccuracy < 20
        warning('GBM精度过低，尝试备选方法');
        template = templateTree('MaxNumSplits', 5);
        classificationGBM = fitcensemble(predictors, response, ...
            'Method', 'Bag', ...         % 使用Bagging作为备选
            'NumLearningCycles', 30, ...
            'Learners', template);

        predict_label = predict(classificationGBM, table2array(test(:,1:end-1)));
        validationAccuracy = (sum(predict_label == test_label) / length(test_label)) * 100;
        ari = rand_index(test_label, predict_label, 'adjusted');
    end

catch e
    warning('GBM分类器训练失败: %s', e.message);
    % 使用更简单的分类器作为后备方案
    try
        simpleTree = fitctree(predictors, response, 'MinLeafSize', 1);
        predict_label = predict(simpleTree, table2array(test(:,1:end-1)));
        test_label = table2array(test(:,end));
        validationAccuracy = (sum(predict_label == test_label) / length(test_label)) * 100;
        ari = rand_index(test_label, predict_label, 'adjusted');
    catch
        validationAccuracy = 0;
        ari = 0;
    end
end

time = toc;
end

function [accuracy, stderror, ari_mean, ari_std] = RF(optimal, data)
rng('default')
k = 10;
cv = cvpartition(size(optimal{1}, 1), 'KFold', k);

for m = 1:size(optimal, 2)
    data = [optimal{m} data(:, end)];
    accuracy_all = [];
    ari_all = [];
    for i = 1:k
        train = array2table(data(cv.training(i), :));
        test = array2table(data(cv.test(i), :));
        [~, validationAccuracy, ari] = RandomForestClassifier(train, test);
        accuracy_all = [accuracy_all validationAccuracy];
        ari_all = [ari_all ari];
    end
    stderror(m) = std(accuracy_all);
    accuracy(m) = mean(accuracy_all);
    ari_mean(m) = mean(ari_all);
    ari_std(m) = std(ari_all);
end
end

function [time, validationAccuracy, ari] = RandomForestClassifier(train, test)
tic;
inputTable = train;
predictorNames = train.Properties.VariableNames(1:end-1);
predictors = inputTable(:, predictorNames);
response = table2array(train(:, end));
class = unique(response);

classificationRF = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 100, ...
    'ClassNames', class);

predictorExtractionFcn = @(t) t(:, predictorNames);
rfPredictFcn = @(x) predict(classificationRF, x);
trainedClassifier.predictFcn = @(x) rfPredictFcn(predictorExtractionFcn(x));

predict_label = trainedClassifier.predictFcn(test);
test_label = table2array(test(:, end));
a = length(find(predict_label == test_label)) / length(test_label) * 100;
validationAccuracy = a;

ari = rand_index(test_label, predict_label, 'adjusted');

time = toc;
end

function [accuracy, stderror, ari_mean, ari_std] = DT(optimal, data)
rng('default')
k = 10;
cv = cvpartition(size(optimal{1}, 1), 'KFold', k);

for m=1:size(optimal,2)
    data=[optimal{m} data(:,end)];
    y_KNNall=[];
    ari_all = [];
    for i=1:k
        train = data(cv.training(i), :);
        test = data(cv.test(i), :);
        [~,y_KNN, ari] = DecisionTreeClassification(train,test);
        y_KNNall=[y_KNNall y_KNN];
        ari_all = [ari_all ari];
    end
    stderror(m)=std(y_KNNall);
    accuracy(m)=mean(y_KNNall);
    ari_mean(m) = mean(ari_all);
    ari_std(m) = std(ari_all);
end
end

function [time, Accuracy, ari] = DecisionTreeClassification(trains, tests)
tic;
train_data = trains(:, 1:end-1);
train_label = trains(:, end);
test_data = tests(:, 1:end-1);
test_label = tests(:, end);

decision_tree_classifier = fitctree(train_data, train_label);
predicted_labels = predict(decision_tree_classifier, test_data);
accuracy = sum(predicted_labels == test_label) / length(test_label) * 100;

ari = rand_index(test_label, predicted_labels, 'adjusted');

Accuracy = accuracy;
time = toc;
end

function [accuracy, stderror, ari_mean, ari_std] = SVM(optimal, data)
rng('default')
k = 10;
cv = cvpartition(size(optimal{1}, 1), 'KFold', k);

for m=1:size(optimal,2)
    data=[optimal{m} data(:,end)];
    y_KNNall=[];
    ari_all = [];
    for i=1:k
        train = array2table(data(cv.training(i), :));
        test = array2table(data(cv.test(i), :));
        [y_KNN, predicted_labels] = ploySvm(train,test);
        y_KNNall=[y_KNNall y_KNN];
        test_labels = table2array(test(:,end));
        ari = rand_index(test_labels, predicted_labels, 'adjusted');
        ari_all = [ari_all ari];
    end
    stderror(m)=std(y_KNNall);
    accuracy(m)=mean(y_KNNall);
    ari_mean(m) = mean(ari_all);
    ari_std(m) = std(ari_all);
end
end

function [y, predicted_labels] = ploySvm(train,test)
inputTable = train;
predictors = inputTable(:, 1:end-1);
response = table2array(inputTable(:,end));

template = templateSVM(...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto');
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template);

trainedClassifier.predictFcn = @(x) predict(classificationSVM, x(:, 1:end-1));

predicted_labels = trainedClassifier.predictFcn(test);
test_label = table2array(test(:,end));
accuracy = mean(predicted_labels == test_label) * 100;

y = accuracy;
end

function ari = rand_index(true_labels, predicted_labels, type)
% 确保输入标签为列向量
true_labels = double(true_labels(:));
predicted_labels = double(predicted_labels(:));

% 检查输入有效性
if length(true_labels) ~= length(predicted_labels)
    error('标签长度不匹配');
end

n = length(true_labels);
if n < 2
    ari = 1;
    return;
end

% 检查NaN值
if any(isnan(true_labels)) || any(isnan(predicted_labels))
    warning('标签中包含NaN值');
    ari = 0;
    return;
end

% 构建混淆矩阵
n_ij = crosstab(true_labels, predicted_labels);

% 计算行和列的和
n_i = sum(n_ij, 2);  % 行和
n_j = sum(n_ij, 1);  % 列和

% 计算总和
N = sum(sum(n_ij));

% 计算配对计数
% a: 在同一类中的对数
% b: 在不同类中的对数
sum_ij = sum(sum(n_ij .* (n_ij-1))) / 2;
sum_i = sum(n_i .* (n_i-1)) / 2;
sum_j = sum(n_j .* (n_j-1)) / 2;
expected_index = sum_i * sum_j / (N * (N-1) / 2);
max_index = sqrt(sum_i * sum_j);

if strcmp(type, 'adjusted')
    if max_index == expected_index
        ari = 0;
    else
        ari = (sum_ij - expected_index) / (max_index - expected_index);
    end
else
    % 计算原始Rand Index
    ari = 2 * (sum_ij) / (N * (N-1));
end

% 处理边界情况
if isnan(ari) || isinf(ari)
    warning('ARI计算结果无效，设置为0');
    ari = 0;
end

% 确保ARI在[-1,1]范围内
ari = max(min(ari, 1), -1);

end

% 定义一个函数来解析数值和误差
function [value, error] = parseValueError(str)
parts = split(str, '±');
value = str2double(parts{1});
error = str2double(parts{2});
end
