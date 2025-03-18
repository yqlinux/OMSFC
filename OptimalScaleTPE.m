function Opt = OptimalScaleTPE(tmpdata, data)

%% MainCodes
% 第一阶段：使用20250307的策略获得好的初始解
initial_contexts = cell(length(tmpdata{1, 1}), 1);
initial_scores = zeros(length(tmpdata{1, 1}), 1);

parfor k = 1:length(tmpdata{1, 1})
    [CV0] = SubFun1(Breakpoint2Context(tmpdata{1,1}{k}, data, k), data);
    OptGra = tmpdata{1,1}{k};
    initial_scores(k) = CV0;
    
    up = tmpdata{1,size(tmpdata,2)}{k};
    for layer = (size(tmpdata,2)):-1:2
        down = tmpdata{1,layer-1}{k};
        Diff1 = setdiff(down, up);
        
        for num = 1:length(Diff1)
            TmpBreakPoint = unique([up Diff1(1:num)]);
            [MRI] = SubFun1(Breakpoint2Context(TmpBreakPoint, data, k), data);
            
            if MRI >= CV0
                OptGra = TmpBreakPoint;
                CV0 = MRI;
            end 
        end 
    end 
    initial_contexts{k} = OptGra;
end

% 第二阶段：构建搜索空间
cuts = {};
count = 1;
for k = 1:length(tmpdata{1, 1})
    base_cut = initial_contexts{k};  % 使用RAOMS找到的解作为基准
    cuts{k,count} = base_cut;
    count = count + 1;
    
    % 向上和向下扩展搜索空间
    up = tmpdata{1,size(tmpdata,2)}{k};
    down = tmpdata{1,1}{k};
    
    % 向下扩展（添加更细粒度）
    diff_down = setdiff(down, base_cut);
    if ~isempty(diff_down)
        for i = 1:min(length(diff_down), 3)  % 限制扩展数量
            temp_cut = unique([base_cut diff_down(1:i)]);
            cuts{k,count} = temp_cut;
            count = count + 1;
        end
    end
    
    % 向上扩展（移除部分节点）
    diff_up = setdiff(base_cut, up);
    if ~isempty(diff_up)
        for i = 1:min(length(diff_up), 3)  % 限制扩展数量
            temp_cut = setdiff(base_cut, diff_up(1:i));
            if length(temp_cut) >= 2  % 确保至少保留两个节点
                cuts{k,count} = temp_cut;
                count = count + 1;
            end
        end
    end
    count = 1;
end

% 构建搜索空间
parameter_space = struct();
for j = 1:size(cuts,1)
    if size(cuts{j},2) <= 2
        continue
    end
    parameter_space.("nodes"+string(j)) = struct(...
        'type', 'categorical',...
        'range', 1:size(cuts{j},2));
end

% 设置TPE优化参数
tpe_params = struct(...
    'n_candidates', 40, ...    % 增加候选点数量
    'gamma', 0.25, ...         % 平衡探索与利用
    'n_startup_trials', 40,... % 初始随机采样数
    'n_trials', 100, ...       % 总试验次数
    'min_bandwidth', 0.001,... % KDE最小带宽
    'adaptive_sampling', true, ... % 启用自适应采样
    'exploitation_weight', 0.8, ... % 增加利用权重
    'early_stopping', true, ... % 启用早停
    'patience', 15, ... % 早停耐心值
    'top_candidates_ratio', 0.2); % 评估的候选点比例

% 执行TPE优化
[best_params, ~] = runTPEOptimization(parameter_space, ...
    @(params) getObjValue(params, cuts, data, initial_scores), ...
    tpe_params);

% 构建最终结果
Datas = [];
for j = 1:size(cuts,1)
    if size(cuts{j},2) <= 2
        continue
    end
    value = best_params.("nodes"+string(j));
    temp = cuts{j};
    y = Breakpoint2Context(temp([1:value length(temp)]), data, j);
    
    % 如果TPE优化的结果不如初始结果，使用初始结果
    current_score = SubFun1(y, data);
    if current_score < initial_scores(j)
        y = Breakpoint2Context(initial_contexts{j}, data, j);
    end
    
    Datas = [Datas y];
end

% 特征选择和属性约简
y = shuxingyuejian(Datas, D2C(data));
diag_elements = diag(y);
Opt = Datas(:, diag_elements ~= 0);

if isempty(Opt) || size(Opt, 2) == 0
    Opt = Datas;
end
end

function [best_params, best_value] = runTPEOptimization(parameter_space, objective_func, tpe_params)
    % TPE优化实现
    n_trials = tpe_params.n_trials;
    n_startup = tpe_params.n_startup_trials;
    gamma = tpe_params.gamma;
    
    % 存储所有试验结果
    trials = struct();
    trials.params = cell(n_trials, 1);
    trials.values = zeros(n_trials, 1);
    
    % 初始随机采样 - 修复parfor问题
    params_array = cell(n_startup, 1);
    values_array = zeros(n_startup, 1);
    
    % 预先获取参数空间信息，避免在parfor中使用fieldnames
    fields_list = fieldnames(parameter_space);
    
    % 使用并行计算进行初始采样
    fprintf('开始初始随机采样 (%d 个点)...\n', n_startup);
    tic;
    parfor i = 1:n_startup
        params = sampleRandomParams(parameter_space, fields_list);
        value = objective_func(params);
        params_array{i} = params;
        values_array(i) = value;
    end
    init_time = toc;
    fprintf('初始采样完成，耗时: %.2f 秒\n', init_time);
    
    % 将结果复制到trials结构体
    for i = 1:n_startup
        trials.params{i} = params_array{i};
        trials.values(i) = values_array(i);
    end
    
    % 记录最佳值和停滞计数
    [best_value_so_far, best_idx] = max(trials.values(1:n_startup));
    best_params_so_far = trials.params{best_idx};
    stagnation_count = 0;
    
    % 创建结果缓存，避免重复计算
    cache = containers.Map();
    
    % TPE迭代优化
    fprintf('开始TPE优化迭代...\n');
    tic;
    for i = (n_startup+1):n_trials
        % 动态调整gamma参数
        if tpe_params.adaptive_sampling
            % 如果优化停滞，增加探索
            if stagnation_count > 10
                current_gamma = max(0.1, gamma * 0.9);  % 增加探索
            else
                current_gamma = min(0.4, gamma * 1.1);  % 增加利用
            end
        else
            current_gamma = gamma;
        end
        
        % 根据gamma参数将观察结果分为好样本和差样本
        [~, sort_idx] = sort(trials.values(1:i-1), 'descend');
        n_good = max(1, round(current_gamma * (i-1)));
        good_idx = sort_idx(1:n_good);
        bad_idx = sort_idx((n_good+1):end);
        
        % 获取好样本和差样本
        good_params = trials.params(good_idx);
        bad_params = trials.params(bad_idx);
        
        % 生成候选点并评估 - 修复parfor问题
        candidates_array = cell(tpe_params.n_candidates, 1);
        ei_values_array = zeros(tpe_params.n_candidates, 1);
        
        % 为每个字段预计算概率分布
        field_probs = cell(length(fields_list), 1);
        for f = 1:length(fields_list)
            field = fields_list{f};
            if strcmp(parameter_space.(field).type, 'categorical')
                range = parameter_space.(field).range;
                probs_good = zeros(1, length(range));
                probs_bad = zeros(1, length(range));
                
                % 统计好样本和差样本中各类别的频率
                for k = 1:length(good_params)
                    val = good_params{k}.(field);
                    probs_good(val) = probs_good(val) + 1;
                end
                for k = 1:length(bad_params)
                    val = bad_params{k}.(field);
                    probs_bad(val) = probs_bad(val) + 1;
                end
                
                % 添加平滑项并归一化
                probs_good = (probs_good + tpe_params.min_bandwidth) / (length(good_params) + tpe_params.min_bandwidth * length(range));
                probs_bad = (probs_bad + tpe_params.min_bandwidth) / (length(bad_params) + tpe_params.min_bandwidth * length(range));
                
                % 计算采样概率
                sample_probs = probs_good ./ (probs_good + probs_bad + eps);
                
                % 应用利用权重
                if isfield(tpe_params, 'exploitation_weight')
                    w = tpe_params.exploitation_weight;
                    uniform_probs = ones(size(sample_probs)) / length(sample_probs);
                    sample_probs = w * sample_probs + (1-w) * uniform_probs;
                end
                
                sample_probs = sample_probs / sum(sample_probs);
                
                % 存储概率分布
                field_probs{f} = struct('field', field, 'range', range, 'probs', sample_probs);
            end
        end
        
        % 生成候选点
        parfor j = 1:tpe_params.n_candidates
            % 为每个参数采样
            candidate = struct();
            
            % 使用预计算的概率分布
            for f = 1:length(field_probs)
                if ~isempty(field_probs{f})
                    field_info = field_probs{f};
                    field = field_info.field;
                    range = field_info.range;
                    sample_probs = field_info.probs;
                    
                    % 根据概率采样
                    candidate.(field) = range(randsample(length(range), 1, true, sample_probs));
                end
            end
            
            candidates_array{j} = candidate;
            
            % 计算EI值
            l_x = computeTPELikelihood(candidate, good_params, parameter_space, fields_list, tpe_params.min_bandwidth);
            g_x = computeTPELikelihood(candidate, bad_params, parameter_space, fields_list, tpe_params.min_bandwidth);
            ei_values_array(j) = l_x / (g_x + eps);
        end
        
        % 将结果复制到本地变量
        candidates = candidates_array;
        candidate_ei = ei_values_array;
        candidate_scores = zeros(tpe_params.n_candidates, 1);
        
        % 评估最有希望的候选点
        [~, ei_idx] = sort(candidate_ei, 'descend');
        top_candidates = max(1, round(tpe_params.n_candidates * tpe_params.top_candidates_ratio));
        
        % 检查缓存并评估候选点
        for j = 1:top_candidates
            idx = ei_idx(j);
            candidate = candidates{idx};
            
            % 创建缓存键
            cache_key = createCacheKey(candidate);
            
            % 检查缓存
            if cache.isKey(cache_key)
                candidate_scores(idx) = cache(cache_key);
            else
                % 评估候选点
                score = objective_func(candidate);
                candidate_scores(idx) = score;
                
                % 更新缓存
                cache(cache_key) = score;
            end
        end
        
        % 选择最佳候选点
        [best_score, best_idx] = max(candidate_scores);
        best_candidate = candidates{best_idx};
        
        % 更新trials
        trials.params{i} = best_candidate;
        trials.values(i) = best_score;
        
        % 检查是否有改进
        if best_score > best_value_so_far
            best_value_so_far = best_score;
            best_params_so_far = best_candidate;
            stagnation_count = 0;
            fprintf('迭代 %d: 发现更好的解，值: %.6f\n', i, best_score);
        else
            stagnation_count = stagnation_count + 1;
        end
        
        % 每10次迭代输出当前最佳值
        if mod(i, 50) == 0
            fprintf('迭代 %d/%d, 当前最佳值: %.6f, 停滞计数: %d\n', i, n_trials, best_value_so_far, stagnation_count);
        end
        
        % 早停检查
        if isfield(tpe_params, 'early_stopping') && tpe_params.early_stopping
            if stagnation_count >= tpe_params.patience
                fprintf('早停触发: %d 次迭代没有改进\n', stagnation_count);
                break;
            end
        end
    end
    
    % 返回最佳结果
    [best_value, best_idx] = max(trials.values);
    best_params = trials.params{best_idx};
    
    % 输出最终结果
    total_time = toc;
    fprintf('优化完成，最佳值: %.6f, 总耗时: %.2f 秒\n', best_value, total_time);
    
    % 如果早停触发，确保返回找到的最佳解
    if best_value < best_value_so_far
        best_value = best_value_so_far;
        best_params = best_params_so_far;
        fprintf('使用早停前找到的最佳解: %.6f\n', best_value);
    end
end

function cache_key = createCacheKey(params)
    % 创建缓存键
    fields = fieldnames(params);
    key_parts = cell(length(fields), 1);
    
    for i = 1:length(fields)
        field = fields{i};
        key_parts{i} = sprintf('%s:%d', field, params.(field));
    end
    
    cache_key = strjoin(key_parts, '_');
end

function likelihood = computeTPELikelihood(candidate, samples, parameter_space, fields_list, min_bandwidth)
    % 计算候选点在样本集中的似然
    if isempty(samples)
        likelihood = min_bandwidth;
        return;
    end
    
    % 计算每个参数的似然
    log_likelihood = 0;
    
    for f = 1:length(fields_list)
        field = fields_list{f};
        if strcmp(parameter_space.(field).type, 'categorical')
            range = parameter_space.(field).range;
            counts = zeros(1, length(range));
            
            % 统计样本中各类别的频率
            for k = 1:length(samples)
                val = samples{k}.(field);
                counts(val) = counts(val) + 1;
            end
            
            % 添加平滑项并计算概率
            probs = (counts + min_bandwidth) / (length(samples) + min_bandwidth * length(range));
            
            % 计算当前参数值的对数似然
            val = candidate.(field);
            log_likelihood = log_likelihood + log(probs(val));
        end
    end
    
    likelihood = exp(log_likelihood);
end

function params = sampleRandomParams(parameter_space, fields_list)
    % 随机采样参数
    params = struct();
    
    for f = 1:length(fields_list)
        field = fields_list{f};
        if strcmp(parameter_space.(field).type, 'categorical')
            range = parameter_space.(field).range;
            params.(field) = range(randi(length(range)));
        end
    end
end