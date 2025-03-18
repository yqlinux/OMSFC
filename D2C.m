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