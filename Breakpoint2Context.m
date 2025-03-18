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