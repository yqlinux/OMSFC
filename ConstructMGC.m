function [Contexts, MS, Breakpoints]=ConstructMGC(data, Breakpoints)

%%
[rows,cols] = size(data); % 获取数据的行数和列数

MS = [];
Contexts = [];
for col = 1:cols        % 遍历每一列
    % 方法1
    dat = data(:,col);         % 获取当前列的数据
    Interval = sort(Breakpoints{col});          % 对当前列的断点进行排序
    [~, C, ~, D] = kmeans(dat, length(Interval)-1);
    % 计算每个簇的点数
    clusterSizes = zeros(length(C), 1); % 初始化每个簇的大小
    for i = 1:length(C)
        clusterSizes(i) = sum(D(:,i) == min(D, [], 2)); % 计算每个簇的数据点数量
    end
    [~, largestClusterIdx] = max(clusterSizes);    % 找到包含最多数据点的簇
    Breakpoints{col} = sort(unique([Breakpoints{col}  C(largestClusterIdx)]));

    % % 方法2
    % max_val = max(Breakpoints{col});
    % min_val = min(Breakpoints{col});
    % step = (max_val - min_val) / (length(Interval));
    % Breakpoints{col} = sort(unique(min_val:step:max_val));

    
    %% 根据断点计算对应的形式背景
    tmppoints = Breakpoints{col};
    Context = zeros(rows, length(tmppoints)-1);  % 预分配空间以提高效率

    for k = 1:(length(tmppoints) - 1)
        if k == length(tmppoints) - 1
            Context(:, k) = (data(:, col) >= tmppoints(k)) & (data(:, col) <= tmppoints(k + 1));
        else
            % 中间的断点保持不变
            Context(:, k) = (data(:, col) >= tmppoints(k)) & (data(:, col) < tmppoints(k + 1));
        end
    end
    Contexts = [Contexts, Context];  % 使用逗号连接而不是逐列追加

    S = 0;
    for ii=1:size(Context,2)
        S = S + Context(:,ii)*ii;
    end
    MS = [MS S];
end

end


