function y = SubFun1(Context, data)
    
    %%
    ConditionGC = unique(Granular_concept(Context), 'rows', 'stable');
    DecisonGC = unique(Granular_concept(D2C(data)), 'rows', 'stable');

    G = size(data, 1);
    epsilon = 1e-10;
    InfoGain = zeros(1, size(ConditionGC, 1));

    for i = 1:size(ConditionGC, 1)
        ObjCC = ConditionGC(i, 1:G);

        CIE = 0;  
        EntS = 0;
        for j = 1:size(DecisonGC, 1)
            ObjDC = DecisonGC(j, 1:G);
            Pf = length(find(ObjDC == 1)) / G;
            EntS = EntS + -Pf * log2(max(Pf, epsilon));

            Pcf = length(find((ObjDC & ObjCC) == 1)) / G;  
            CIE = CIE + -Pcf * log2(max(length(find((ObjDC & ObjCC)==1)) / length(find(ObjCC==1)), epsilon));
        end

        EntS = EntS / size(DecisonGC, 1);
        InfoGain(i) = EntS - CIE;
    end

    y = mean(InfoGain);
    
end