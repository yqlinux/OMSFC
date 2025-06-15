function [Contexts, MS, Breakpoints]=ConstructMGC(data, Breakpoints)

%%
[rows,cols] = size(data);

MS = [];
Contexts = [];
for col = 1:cols
    dat = data(:,col);
    Interval = sort(Breakpoints{col});
    [~, C, ~, D] = kmeans(dat, length(Interval)-1);
    clusterSizes = zeros(length(C), 1);
    for i = 1:length(C)
        clusterSizes(i) = sum(D(:,i) == min(D, [], 2));
    end
    [~, largestClusterIdx] = max(clusterSizes);
    Breakpoints{col} = sort(unique([Breakpoints{col}  C(largestClusterIdx)]));

    tmppoints = Breakpoints{col};
    Context = zeros(rows, length(tmppoints)-1);

    for k = 1:(length(tmppoints) - 1)
        if k == length(tmppoints) - 1
            Context(:, k) = (data(:, col) >= tmppoints(k)) & (data(:, col) <= tmppoints(k + 1));
        else
            Context(:, k) = (data(:, col) >= tmppoints(k)) & (data(:, col) < tmppoints(k + 1));
        end
    end
    Contexts = [Contexts, Context];

    S = 0;
    for ii=1:size(Context,2)
        S = S + Context(:,ii)*ii;
    end
    MS = [MS S];
end

end