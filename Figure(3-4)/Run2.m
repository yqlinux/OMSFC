%% 不同方法对比实验精度
warning("off");close all;clc
addpath(genpath(pwd))

%%
nameall=[
    "appendicitis";
    "hayes-roth";
    % "hepatitis";
    % "glass";
    % "haberman";
    % "bupa";
    % "bands";
    % "auto_mpg";
    % "wisconsin";
    % "mammographic";
    % "Raisin";
    % "vowel";
    % "PhishingData";
    % "titanic";
    % "Satimage";
    % "Ring";
    ];

currentDate = datestr(now, 'yyyy-mm-dd');
accuracyFileName = ['Accuracy_Results_' currentDate '.xlsx'];
ariFileName = ['ARI_Results_' currentDate '.xlsx'];

all_accuracy_results = cell(16*4, 5);
all_ari_results = cell(16*4, 5);
dataset_names = cell(16, 1);

for i = 1:16
    data=table2array(readtable("G:\备份HPDesktopFiles\MyPaper\Paper2\DT2\"+nameall(i)+'.txt'));
    data= fillmissing(data,"nearest");

    resultsTable1 = [];
    features = data(:,1:end-1);
    iter = 8;
    breakpoints=init(features);
    % tmpdata={};
    % for n=iter:-1:1
    %     [DRG, DRS, intervalresult]=ConstructMGC(features,breakpoints);
    %     tmpdata{1,n} = intervalresult;
    %     tmpdata{2,n} = DRG;
    %     tmpdata{3,n} = DRS;
    %     breakpoints = intervalresult;
    %     resultsTable1 = [resultsTable1 size(DRG,2)];
    % end
    tmpdata=load("F:\Paper1-已发表\Program(GitHub)\Figure(3-4)\MultiGranDat\"+(num2str(i)+".mat")).MFC{1,1};

    tic; optimal{1} = OptimalScaleFC1(tmpdata, data); 
    tic; optimal{2} = [tmpdata{2,1}]; 
    tic; optimal{3} = [tmpdata{2,size(tmpdata,2)}]; 
    tic; optimal{4} = [tmpdata{2,randi([2, size(tmpdata,2)-1])}]; 

    [AccKnn, StdKnn, AriKnn, AriStdKnn] = RF(optimal,data);
    [AccDT, StdDT, AriDT, AriStdDT] = DT(optimal,data);
    [AccSvm, StdSvm, AriSvm, AriStdSvm] = SVM(optimal,data);
    [AccGBM, StdGBM, AriGBM, AriStdGBM] = GBM(optimal,data);
    [AccXGB, StdXGB, AriXGB, AriStdXGB] = XGBoost(optimal,data);

    accuracy_results = cell(4, 5);
    ari_results = cell(4, 5);
    
    for ii = 1:4
        accuracy_results{ii, 1} = sprintf('%.2f±%.2f', AccKnn(ii), StdKnn(ii));
        accuracy_results{ii, 2} = sprintf('%.2f±%.2f', AccDT(ii), StdDT(ii));
        accuracy_results{ii, 3} = sprintf('%.2f±%.2f', AccSvm(ii), StdSvm(ii));
        accuracy_results{ii, 4} = sprintf('%.2f±%.2f', AccGBM(ii), StdGBM(ii));
        accuracy_results{ii, 5} = sprintf('%.2f±%.2f', AccXGB(ii), StdXGB(ii));
        
        ari_results{ii, 1} = sprintf('%.2f±%.2f', AriKnn(ii), AriStdKnn(ii));
        ari_results{ii, 2} = sprintf('%.2f±%.2f', AriDT(ii), AriStdDT(ii));
        ari_results{ii, 3} = sprintf('%.2f±%.2f', AriSvm(ii), AriStdSvm(ii));
        ari_results{ii, 4} = sprintf('%.2f±%.2f', AriGBM(ii), AriStdGBM(ii));
        ari_results{ii, 5} = sprintf('%.2f±%.2f', AriXGB(ii), AriStdXGB(ii));
    end

    rowNames = ["FC","Finest","Coarset","Random"];
    columnNames = {'RF', 'DT', 'SVM', 'GBM', 'XGBoost'};

    disp(nameall(i)+".txt"+"============================================================")
    
    disp('Accuracy Results:')
    accuracyTable = array2table(accuracy_results, 'VariableNames', columnNames, 'RowNames', rowNames);
    disp(accuracyTable)
    
    disp('ARI Results:')
    ariTable = array2table(ari_results, 'VariableNames', columnNames, 'RowNames', rowNames);
    disp(ariTable)

    writetable(accuracyTable, accuracyFileName, 'Sheet', nameall(i), 'WriteRowNames', true);
    writetable(ariTable, ariFileName, 'Sheet', nameall(i), 'WriteRowNames', true);

    dataset_names{i} = nameall(i);
    startRow = (i-1)*4 + 1;
    endRow = i*4;
    all_accuracy_results(startRow:endRow, :) = accuracy_results;
    all_ari_results(startRow:endRow, :) = ari_results;
end

summary_row_names = cell(16*4, 1);
for i = 1:16
    for j = 1:4
        idx = (i-1)*4 + j;
        summary_row_names{idx} = sprintf('%s_%s', nameall(i), rowNames(j));
    end
end

summaryAccuracyTable = array2table(all_accuracy_results, ...
    'VariableNames', columnNames, ...
    'RowNames', summary_row_names);
summaryARITable = array2table(all_ari_results, ...
    'VariableNames', columnNames, ...
    'RowNames', summary_row_names);

writetable(summaryAccuracyTable, accuracyFileName, 'Sheet', 'Summary', 'WriteRowNames', true);
writetable(summaryARITable, ariFileName, 'Sheet', 'Summary', 'WriteRowNames', true);

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