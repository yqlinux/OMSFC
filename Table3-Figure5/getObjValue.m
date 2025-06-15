function objValue = getObjValue(parameter, cuts, data, initial_scores)

%%
Datas = [];
total_score = 0;
valid_features = 0;

for j = 1:size(cuts,1)
    if size(cuts{j},2) <= 2
        continue
    end
    
    value = parameter.("nodes"+string(j));
    temp = cuts{j};
    current_cut = temp([1:double(string(value)) length(temp)]);
    
    current_context = Breakpoint2Context(current_cut, data, j);
    current_score = SubFun1(current_context, data);
    
    if current_score >= initial_scores(j)
        Datas = [Datas current_context];
        total_score = total_score + current_score;
        valid_features = valid_features + 1;
    end
end

if isempty(Datas)
    objValue = -inf;
    return;
end

ensemble_score = SubFun1(Datas, data);

if valid_features > 0
    avg_improvement = total_score / valid_features;
else
    avg_improvement = 0;
end

objValue = ensemble_score * (1 + 0.2 * avg_improvement);

end
