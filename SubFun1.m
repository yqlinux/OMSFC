function y = SubFun1(Context, data)
    ConditionGC = unique(Granular_concept(Context), 'rows', 'stable');
    DecisonGC = unique(Granular_concept(D2C(data)), 'rows', 'stable');

    G = size(data, 1);  % 样本数量
    epsilon = 1e-10;    % 防止对数运算出现NaN
    InfoGain = zeros(1, size(ConditionGC, 1));    % 初始化信息增益数组

    % 计算每个粒度概念的信息增益
    for i = 1:size(ConditionGC, 1)
        ObjCC = ConditionGC(i, 1:G);  % 条件粒概念外延
        % PC = length(find(ObjCC == 1)) / G;  % 条件粒度概念的概率 p(f)

        CIE = 0;  
        EntS = 0; % 信息熵
        for j = 1:size(DecisonGC, 1)
            ObjDC = DecisonGC(j, 1:G);  % 决策粒概念外延
            Pf = length(find(ObjDC == 1)) / G;  % 决策粒度概念的概率 p(f)
            EntS = EntS + -Pf * log2(max(Pf, epsilon));% 计算决策的熵

            Pcf = length(find((ObjDC & ObjCC) == 1)) / G;  
            CIE = CIE + -Pcf * log2(max(length(find((ObjDC & ObjCC)==1)) / length(find(ObjCC==1)), epsilon));
        end

        EntS = EntS / size(DecisonGC, 1);
        InfoGain(i) = EntS - CIE;% 计算信息增益
    end

    y = mean(InfoGain);
end