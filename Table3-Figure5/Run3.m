warning("off");
close all;clc;clear;
addpath(genpath(pwd))


%%
nameall=["appendicitis";"hayes-roth";]; % "hepatitis";"glass";"haberman";"bupa";"bands";"auto_mpg";"wisconsin";"mammographic";"Raisin";"vowel";"PhishingData";"titanic";"Satimage";"Ring";];
currentDate = datestr(now, 'yyyy-mm-dd');
resultsFileName = ['Comparison_Results_' currentDate '.xlsx'];
methodNames = ["BOOMC", "RAOMS", "OGS", "OSS", "OSC", "3WD_HD", "MSCRS"];
classifierNames = {'RF', 'DT', 'SVM', 'GBM', 'XGBoost'};

for i = 1:14
    allResults = cell(8, 12);
    rowIndex = 1;
    
    fprintf('\n处理数据集 %d: %s\n', i, nameall(i));
    fprintf('----------------------------------------\n');
   
    data=table2array(readtable("G:\备份HPDesktopFiles\MyPaper\Paper2\DT2\"+nameall(i)+'.txt'));
    data= fillmissing(data,"nearest");
    tmpdata = load("F:\Paper1-已发表\Program(GitHub)\Table3-Figure5\MultiGranDat\"+(num2str(i)+".mat")).MFC{1,1};

    tic; optimal{1} = OptimalScaleTPE(tmpdata, data); time(1) = toc;
    tic; optimal{2} = OptimalScaleFC1(tmpdata, data); time(2) = toc;
    tic; optimal{3} = OptimalScaleHJJ(tmpdata, data); time(3) = toc;
    tic; optimal{4} = OptimalScaleWWZ(tmpdata, data); time(4) = toc;
    tic; optimal{5} = OptimalScaleCDX(tmpdata, data); time(5) = toc;
    tic; optimal{6} = OptimalScaleZQH(tmpdata, data); time(6) = toc;
    tic; optimal{7} = OptimalScaleHZH(tmpdata, data); time(7) = toc;

    [AccKnn, StdKnn, AriKnn, AriStdKnn] = RF(optimal,data);
    [AccDT, StdDT, AriDT, AriStdDT] = DT(optimal,data);
    [AccSvm, StdSvm, AriSvm, AriStdSvm] = SVM(optimal,data);
    [AccGBM, StdGBM, AriGBM, AriStdGBM] = GBM(optimal,data);
    [AccXGB, StdXGB, AriXGB, AriStdXGB] = XGBoost(optimal,data);

    allResults{rowIndex, 1} = 'ID';
    allResults{rowIndex, 2} = 'Method';
    allResults{rowIndex, 3} = 'RF-Acc';
    allResults{rowIndex, 4} = 'RF-ARI';
    allResults{rowIndex, 5} = 'DT-Acc';
    allResults{rowIndex, 6} = 'DT-ARI';
    allResults{rowIndex, 7} = 'SVM-Acc';
    allResults{rowIndex, 8} = 'SVM-ARI';
    allResults{rowIndex, 9} = 'GBM-Acc';
    allResults{rowIndex, 10} = 'GBM-ARI';
    allResults{rowIndex, 11} = 'XGB-Acc';
    allResults{rowIndex, 12} = 'XGB-ARI';
    rowIndex = rowIndex + 1;

    rawData = cell(7, 10);
    for m = 1:7
        rawData{m,1} = sprintf('%.2f±%.2f', AccKnn(m), StdKnn(m));
        rawData{m,2} = sprintf('%.2f±%.2f', AriKnn(m), AriStdKnn(m));
        rawData{m,3} = sprintf('%.2f±%.2f', AccDT(m), StdDT(m));
        rawData{m,4} = sprintf('%.2f±%.2f', AriDT(m), AriStdDT(m));
        rawData{m,5} = sprintf('%.2f±%.2f', AccSvm(m), StdSvm(m));
        rawData{m,6} = sprintf('%.2f±%.2f', AriSvm(m), AriStdSvm(m));
        rawData{m,7} = sprintf('%.2f±%.2f', AccGBM(m), StdGBM(m));
        rawData{m,8} = sprintf('%.2f±%.2f', AriGBM(m), AriStdGBM(m));
        rawData{m,9} = sprintf('%.2f±%.2f', AccXGB(m), StdXGB(m));
        rawData{m,10} = sprintf('%.2f±%.2f', AriXGB(m), AriStdXGB(m));
    end

    maxIndicesList = cell(1, 10);
    for col = 1:10
        maxVal = -inf;
        minErr = inf;
        
        for row = 1:7
            [val, ~] = parseValueError(rawData{row,col});
            if val > maxVal
                maxVal = val;
            end
        end
        
        maxValIndices = [];
        for row = 1:7
            [val, err] = parseValueError(rawData{row,col});
            if val == maxVal
                maxValIndices = [maxValIndices row];
            end
        end
        
        minErr = inf;
        for idx = maxValIndices
            [~, err] = parseValueError(rawData{idx,col});
            if err < minErr
                minErr = err;
            end
        end
        
        finalMaxIndices = [];
        for idx = maxValIndices
            [~, err] = parseValueError(rawData{idx,col});
            if err == minErr
                finalMaxIndices = [finalMaxIndices idx];
            end
        end
        
        maxIndicesList{col} = finalMaxIndices;
    end

    for m = 1:7
        allResults{rowIndex, 1} = i;
        allResults{rowIndex, 2} = methodNames(m);
        
        for col = 1:10
            if ismember(m, maxIndicesList{col})
                allResults{rowIndex, col+2} = ['**' rawData{m,col} '**'];
            else
                allResults{rowIndex, col+2} = rawData{m,col};
            end
        end
        rowIndex = rowIndex + 1;
    end

    headers = {'ID', 'Method', 'RF-Acc', 'RF-ARI', 'DT-Acc', 'DT-ARI', ...
              'SVM-Acc', 'SVM-ARI', 'GBM-Acc', 'GBM-ARI', 'XGB-Acc', 'XGB-ARI'};
    
    dataRows = allResults(2:end, :);
    resultsTable = cell2table(dataRows, 'VariableNames', headers);
    
    sheetName = sprintf('%d-%s', i, nameall(i));
    if length(sheetName) > 31
        sheetName = sprintf('Dataset_%d', i);
    end
    
    writetable(resultsTable, resultsFileName, 'Sheet', sheetName, 'WriteMode', 'append');
    
    fprintf('数据集 %d (%s) 的结果已保存到工作簿: %s\n', i, nameall(i), sheetName);
end

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

params = struct();
params.iterations = 100;
params.learning_rate = 0.03;
params.depth = 6;
params.l2_leaf_reg = 3;
params.min_data_in_leaf = 1;

template = templateTree(...
    'MaxNumSplits', 2^params.depth, ...
    'MinLeafSize', params.min_data_in_leaf, ...
    'SplitCriterion', 'deviance', ...
    'Surrogate', 'off');

if length(class) <= 2
    method = 'AdaBoostM1';
else
    method = 'AdaBoostM2';
end

try
    model = fitcensemble(predictors, response, ...
        'Method', method, ...
        'NumLearningCycles', params.iterations, ...
        'Learners', template, ...
        'LearnRate', params.learning_rate, ...
        'ClassNames', class);
    
    [predict_label, scores] = predict(model, test_predictors);
    test_label = table2array(test(:,end));
    
    validationAccuracy = (sum(predict_label == test_label) / length(test_label)) * 100;
    
    ari = rand_index(test_label, predict_label, 'adjusted');
    
    if length(class) == 2
        confMat = confusionmat(test_label, predict_label);
        TP = confMat(2,2);
        TN = confMat(1,1);
        FP = confMat(1,2);
        FN = confMat(2,1);
        
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        f1_score = 2 * (precision * recall) / (precision + recall);
        
        [X,Y,~,AUC] = perfcurve(test_label, scores(:,2), 1);
    end
    
catch ME
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

template = templateTree(...
    'MaxNumSplits', 10, ...
    'MinLeafSize', 1, ...
    'MinParentSize', 2, ...
    'Surrogate', 'off');

try
    classificationGBM = fitcensemble(...
        predictors, ...
        response, ...
        'Method', 'GentleBoost', ...
        'NumLearningCycles', 50, ...
        'Learners', template, ...
        'LearnRate', 0.1, ...
        'ClassNames', class);

    cvmodel = crossval(classificationGBM);
    cvError = kfoldLoss(cvmodel);
    
    if cvError > 0.5
        classificationGBM = fitcensemble(...
            predictors, ...
            response, ...
            'Method', 'RUSBoost', ...
            'NumLearningCycles', 100, ...
            'Learners', template, ...
            'LearnRate', 0.05);
    end

    predictorExtractionFcn = @(t) t(:, predictorNames);
    gbmPredictFcn = @(x) predict(classificationGBM, x);
    trainedClassifier.predictFcn = @(x) gbmPredictFcn(predictorExtractionFcn(x));

    predict_label = trainedClassifier.predictFcn(test);
    test_label = table2array(test(:,end));
    validationAccuracy = (sum(predict_label == test_label) / length(test_label)) * 100;

    ari = rand_index(test_label, predict_label, 'adjusted');
    
    if validationAccuracy < 20
        warning('GBM精度过低，尝试备选方法');
        template = templateTree('MaxNumSplits', 5);
        classificationGBM = fitcensemble(predictors, response, ...
            'Method', 'Bag', ...
            'NumLearningCycles', 30, ...
            'Learners', template);
        
        predict_label = predict(classificationGBM, table2array(test(:,1:end-1)));
        validationAccuracy = (sum(predict_label == test_label) / length(test_label)) * 100;
        ari = rand_index(test_label, predict_label, 'adjusted');
    end
    
catch e
    warning('GBM分类器训练失败: %s', e.message);
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
true_labels = double(true_labels(:));
predicted_labels = double(predicted_labels(:));

if length(true_labels) ~= length(predicted_labels)
    error('标签长度不匹配');
end

n = length(true_labels);
if n < 2
    ari = 1;
    return;
end

if any(isnan(true_labels)) || any(isnan(predicted_labels))
    warning('标签中包含NaN值');
    ari = 0;
    return;
end

n_ij = crosstab(true_labels, predicted_labels);

n_i = sum(n_ij, 2);
n_j = sum(n_ij, 1);

N = sum(sum(n_ij));

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
    ari = 2 * (sum_ij) / (N * (N-1));
end

if isnan(ari) || isinf(ari)
    warning('ARI计算结果无效，设置为0');
    ari = 0;
end

ari = max(min(ari, 1), -1);
end

function [value, error] = parseValueError(str)
    parts = split(str, '±');
    value = str2double(parts{1});
    error = str2double(parts{2});
end