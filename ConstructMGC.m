function [Contexts, MS, Breakpoints]=ConstructMGC(data, Breakpoints)

%%
[rows,cols] = size(data); % ��ȡ���ݵ�����������

MS = [];
Contexts = [];
for col = 1:cols        % ����ÿһ��
    % ����1
    dat = data(:,col);         % ��ȡ��ǰ�е�����
    Interval = sort(Breakpoints{col});          % �Ե�ǰ�еĶϵ��������
    [~, C, ~, D] = kmeans(dat, length(Interval)-1);
    % ����ÿ���صĵ���
    clusterSizes = zeros(length(C), 1); % ��ʼ��ÿ���صĴ�С
    for i = 1:length(C)
        clusterSizes(i) = sum(D(:,i) == min(D, [], 2)); % ����ÿ���ص����ݵ�����
    end
    [~, largestClusterIdx] = max(clusterSizes);    % �ҵ�����������ݵ�Ĵ�
    Breakpoints{col} = sort(unique([Breakpoints{col}  C(largestClusterIdx)]));

    % % ����2
    % max_val = max(Breakpoints{col});
    % min_val = min(Breakpoints{col});
    % step = (max_val - min_val) / (length(Interval));
    % Breakpoints{col} = sort(unique(min_val:step:max_val));

    
    %% ���ݶϵ�����Ӧ����ʽ����
    tmppoints = Breakpoints{col};
    Context = zeros(rows, length(tmppoints)-1);  % Ԥ����ռ������Ч��

    for k = 1:(length(tmppoints) - 1)
        if k == length(tmppoints) - 1
            Context(:, k) = (data(:, col) >= tmppoints(k)) & (data(:, col) <= tmppoints(k + 1));
        else
            % �м�Ķϵ㱣�ֲ���
            Context(:, k) = (data(:, col) >= tmppoints(k)) & (data(:, col) < tmppoints(k + 1));
        end
    end
    Contexts = [Contexts, Context];  % ʹ�ö������Ӷ���������׷��

    S = 0;
    for ii=1:size(Context,2)
        S = S + Context(:,ii)*ii;
    end
    MS = [MS S];
end

end


