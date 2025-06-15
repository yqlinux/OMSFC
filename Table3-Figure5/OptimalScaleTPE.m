function Opt = OptimalScaleTPE(tmpdata, data)

%%
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

cuts = {};
count = 1;
for k = 1:length(tmpdata{1, 1})
    base_cut = initial_contexts{k};
    cuts{k,count} = base_cut;
    count = count + 1;
    
    up = tmpdata{1,size(tmpdata,2)}{k};
    down = tmpdata{1,1}{k};
    
    diff_down = setdiff(down, base_cut);
    if ~isempty(diff_down)
        for i = 1:min(length(diff_down), 3)
            temp_cut = unique([base_cut diff_down(1:i)]);
            cuts{k,count} = temp_cut;
            count = count + 1;
        end
    end
    
    diff_up = setdiff(base_cut, up);
    if ~isempty(diff_up)
        for i = 1:min(length(diff_up), 3)
            temp_cut = setdiff(base_cut, diff_up(1:i));
            if length(temp_cut) >= 2
                cuts{k,count} = temp_cut;
                count = count + 1;
            end
        end
    end
    count = 1;
end

parameter_space = struct();
for j = 1:size(cuts,1)
    if size(cuts{j},2) <= 2
        continue
    end
    parameter_space.("nodes"+string(j)) = struct(...
        'type', 'categorical',...
        'range', 1:size(cuts{j},2));
end

tpe_params = struct(...
    'n_candidates', 40, ...
    'gamma', 0.25, ...
    'n_startup_trials', 40,...
    'n_trials', 100, ...
    'min_bandwidth', 0.001,...
    'adaptive_sampling', true, ...
    'exploitation_weight', 0.8, ...
    'early_stopping', true, ...
    'patience', 15, ...
    'top_candidates_ratio', 0.2);

[best_params, ~] = runTPEOptimization(parameter_space, ...
    @(params) getObjValue(params, cuts, data, initial_scores), ...
    tpe_params);

Datas = [];
for j = 1:size(cuts,1)
    if size(cuts{j},2) <= 2
        continue
    end
    value = best_params.("nodes"+string(j));
    temp = cuts{j};
    y = Breakpoint2Context(temp([1:value length(temp)]), data, j);
    
    current_score = SubFun1(y, data);
    if current_score < initial_scores(j)
        y = Breakpoint2Context(initial_contexts{j}, data, j);
    end
    
    Datas = [Datas y];
end

y = shuxingyuejian(Datas, D2C(data));
diag_elements = diag(y);
Opt = Datas(:, diag_elements ~= 0);

if isempty(Opt) || size(Opt, 2) == 0
    Opt = Datas;
end

end


%% SubFunctions
function [best_params, best_value] = runTPEOptimization(parameter_space, objective_func, tpe_params)
n_trials = tpe_params.n_trials;
n_startup = tpe_params.n_startup_trials;
gamma = tpe_params.gamma;

trials = struct();
trials.params = cell(n_trials, 1);
trials.values = zeros(n_trials, 1);

params_array = cell(n_startup, 1);
values_array = zeros(n_startup, 1);

fields_list = fieldnames(parameter_space);

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

for i = 1:n_startup
    trials.params{i} = params_array{i};
    trials.values(i) = values_array(i);
end

[best_value_so_far, best_idx] = max(trials.values(1:n_startup));
best_params_so_far = trials.params{best_idx};
stagnation_count = 0;

cache = containers.Map();

fprintf('开始TPE优化迭代...\n');
tic;
for i = (n_startup+1):n_trials
    if tpe_params.adaptive_sampling
        if stagnation_count > 10
            current_gamma = max(0.1, gamma * 0.9);
        else
            current_gamma = min(0.4, gamma * 1.1);
        end
    else
        current_gamma = gamma;
    end
    
    [~, sort_idx] = sort(trials.values(1:i-1), 'descend');
    n_good = max(1, round(current_gamma * (i-1)));
    good_idx = sort_idx(1:n_good);
    bad_idx = sort_idx((n_good+1):end);
    
    good_params = trials.params(good_idx);
    bad_params = trials.params(bad_idx);
    
    candidates_array = cell(tpe_params.n_candidates, 1);
    ei_values_array = zeros(tpe_params.n_candidates, 1);
    
    field_probs = cell(length(fields_list), 1);
    for f = 1:length(fields_list)
        field = fields_list{f};
        if strcmp(parameter_space.(field).type, 'categorical')
            range = parameter_space.(field).range;
            probs_good = zeros(1, length(range));
            probs_bad = zeros(1, length(range));
            
            for k = 1:length(good_params)
                val = good_params{k}.(field);
                probs_good(val) = probs_good(val) + 1;
            end
            for k = 1:length(bad_params)
                val = bad_params{k}.(field);
                probs_bad(val) = probs_bad(val) + 1;
            end
            
            probs_good = (probs_good + tpe_params.min_bandwidth) / (length(good_params) + tpe_params.min_bandwidth * length(range));
            probs_bad = (probs_bad + tpe_params.min_bandwidth) / (length(bad_params) + tpe_params.min_bandwidth * length(range));
            
            sample_probs = probs_good ./ (probs_good + probs_bad + eps);
            
            if isfield(tpe_params, 'exploitation_weight')
                w = tpe_params.exploitation_weight;
                uniform_probs = ones(size(sample_probs)) / length(sample_probs);
                sample_probs = w * sample_probs + (1-w) * uniform_probs;
            end
            
            sample_probs = sample_probs / sum(sample_probs);
            
            field_probs{f} = struct('field', field, 'range', range, 'probs', sample_probs);
        end
    end
    
    parfor j = 1:tpe_params.n_candidates
        candidate = struct();
        
        for f = 1:length(field_probs)
            if ~isempty(field_probs{f})
                field_info = field_probs{f};
                field = field_info.field;
                range = field_info.range;
                sample_probs = field_info.probs;
                
                candidate.(field) = range(randsample(length(range), 1, true, sample_probs));
            end
        end
        
        candidates_array{j} = candidate;
        
        l_x = computeTPELikelihood(candidate, good_params, parameter_space, fields_list, tpe_params.min_bandwidth);
        g_x = computeTPELikelihood(candidate, bad_params, parameter_space, fields_list, tpe_params.min_bandwidth);
        ei_values_array(j) = l_x / (g_x + eps);
    end
    
    candidates = candidates_array;
    candidate_ei = ei_values_array;
    candidate_scores = zeros(tpe_params.n_candidates, 1);
    
    [~, ei_idx] = sort(candidate_ei, 'descend');
    top_candidates = max(1, round(tpe_params.n_candidates * tpe_params.top_candidates_ratio));
    
    for j = 1:top_candidates
        idx = ei_idx(j);
        candidate = candidates{idx};
        
        cache_key = createCacheKey(candidate);
        
        if cache.isKey(cache_key)
            candidate_scores(idx) = cache(cache_key);
        else
            score = objective_func(candidate);
            candidate_scores(idx) = score;
            
            cache(cache_key) = score;
        end
    end
    
    [best_score, best_idx] = max(candidate_scores);
    best_candidate = candidates{best_idx};
    
    trials.params{i} = best_candidate;
    trials.values(i) = best_score;
    
    if best_score > best_value_so_far
        best_value_so_far = best_score;
        best_params_so_far = best_candidate;
        stagnation_count = 0;
        fprintf('迭代 %d: 发现更好的解，值: %.6f\n', i, best_score);
    else
        stagnation_count = stagnation_count + 1;
    end
    
    if mod(i, 50) == 0
        fprintf('迭代 %d/%d, 当前最佳值: %.6f, 停滞计数: %d\n', i, n_trials, best_value_so_far, stagnation_count);
    end
    
    if isfield(tpe_params, 'early_stopping') && tpe_params.early_stopping
        if stagnation_count >= tpe_params.patience
            fprintf('早停触发: %d 次迭代没有改进\n', stagnation_count);
            break;
        end
    end
end

[best_value, best_idx] = max(trials.values);
best_params = trials.params{best_idx};

total_time = toc;
fprintf('优化完成，最佳值: %.6f, 总耗时: %.2f 秒\n', best_value, total_time);

if best_value < best_value_so_far
    best_value = best_value_so_far;
    best_params = best_params_so_far;
    fprintf('使用早停前找到的最佳解: %.6f\n', best_value);
end
end

function cache_key = createCacheKey(params)
fields = fieldnames(params);
key_parts = cell(length(fields), 1);

for i = 1:length(fields)
    field = fields{i};
    key_parts{i} = sprintf('%s:%d', field, params.(field));
end

cache_key = strjoin(key_parts, '_');
end

function likelihood = computeTPELikelihood(candidate, samples, parameter_space, fields_list, min_bandwidth)
if isempty(samples)
    likelihood = min_bandwidth;
    return;
end

log_likelihood = 0;

for f = 1:length(fields_list)
    field = fields_list{f};
    if strcmp(parameter_space.(field).type, 'categorical')
        range = parameter_space.(field).range;
        counts = zeros(1, length(range));
        
        for k = 1:length(samples)
            val = samples{k}.(field);
            counts(val) = counts(val) + 1;
        end
        
        probs = (counts + min_bandwidth) / (length(samples) + min_bandwidth * length(range));
        
        val = candidate.(field);
        log_likelihood = log_likelihood + log(probs(val));
    end
end

likelihood = exp(log_likelihood);
end

function params = sampleRandomParams(parameter_space, fields_list)
params = struct();

for f = 1:length(fields_list)
    field = fields_list{f};
    if strcmp(parameter_space.(field).type, 'categorical')
        range = parameter_space.(field).range;
        params.(field) = range(randi(length(range)));
    end
end
end