function optimal=OptimalScaleCDX(tmpdata,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/13
% @Parameter Instruction：
% reference：[1] DONGXIAO CHEN , Li J , Lin R , et al. Information Entropy and Optimal Scale Combination in Multi-Scale Covering Decision Systems[J]. IEEE Access, 2020, 8:182908-182917.


%% 多粒度形式背景反向尺度化为多尺度信息系统
MT0=tmpdata{2,1};
MT1=tmpdata{1,1};
for i=1:length(MT1)
    ISC(i)=length(MT1{i})-1;
end

% 反向尺度化《形式概念分析的多粒度标记理论》
ind=[0 ISC(1:end-1)];
First_Ind=cumsum(ind);
MT1=zeros(size(data,1),sum(ISC));
for ii=1:length(ISC)
    MT1(:,1+First_Ind(ii))=MT0(:,1+First_Ind(ii));
    for jj=2:ISC(ii)
        MT1(:,jj+First_Ind(ii))=MT1(:,jj-1+First_Ind(ii))+MT0(:,jj-1+First_Ind(ii))*jj;
    end
end
MT0=MT1;

D=data(:,end);
NewData={};
for i=1:min(ISC) 
    Temp=[];
    for j=1:length(ind)
        Temp=[Temp MT0(:,ISC(j)+First_Ind(j))];
    end
    NewData{2,i}=Temp;
end

% NewData = tmpdata;
if IE3(D,(NewData{2,1}'),1) == 0
    optimal = ConsitentOpt(D,NewData);
else
    optimal = InconsitentOpt(D,NewData);
end
end


function optimals=ConsitentOpt(D,data) % Algorithm 1 The Optimal Scale Combination of a Multi-Scale Covering Decision System
% step3
label=0; % IE3计算的都是条件信息熵
if IE3(D,(data{2,end}'),label) == 0
    optimals=data{2,end};
elseif IE3(D,(data{2,end}'),label) > 0
    %% 使用深度优先和广度优先搜索算法复杂度相同，不同的数据集，效率可能不相同。改进搜索算法为贝叶斯搜索，提高搜索效率。
    % step4-> step6
    K=data{2,end};% 初始化K为最粗粒度多粒度形式背景
    layer=size(data,2);% 最粗粒度层层数
    while IE3(D,cell2mat(K'),label) > 0 && layer-1>0 % H({d}|Δ^K) > 0
        Kj=data{2,layer-1};
        for j=1:size(K,1)% 替换属性j的粒度层lj为lj-1
            % 计算最小的条件信息熵和对应的单粒度形式背景，最小条件信息熵多个时选择第一个最小。
            if j==1
                K{j}=Kj{j};
                tmpIE=IE3(D,(K'),label);
                tmpK=K;
            else
                if tmpIE > IE3(D,(K'),label)
                    K{j}=Kj{j};
                    tmpIE=IE3(D,(K'),label);
                    tmpK=K;
                end
            end
        end
        
        layer=layer-1;% 粒度层逐次细化
    end
    optimals=K;
else
    optimals=data{2,end};
end
end

function optimals=InconsitentOpt(D,data)%Algorithm 2 The Optimal Scale Combination of a Inconsistent Multi-Scale Covering Decision System
% step3
label=0; % IE3计算的都是条件信息熵
if IE3(D,(data{2,end}'),label) <= 0
    optimals=data{2,end};
elseif IE3(D,(data{2,end}'),label) > 0
    %% 使用深度优先和广度优先搜索算法复杂度相同，不同的数据集，效率可能不相同。改进搜索算法为贝叶斯搜索，提高搜索效率。
    % step4-> step6
    K=data{2,end};% 初始化K为最粗粒度多粒度形式背景
    layer=size(data,2);% 最粗粒度层层数
    while LH(D,(K'),label,length(POS(D,K'))) > 0 && layer-1>0 % H({d}|Δ^K) > 0，深度搜索的思路。
        U_k0=length(POS(D,K'));%U_k = Pos_(Δ^k) ({d}),即Multi-scale covering rough sets with applications to data classification 一文中的决策d的正域
        Kj=data{2,layer-1};
        for j=1:size(K,2)% 替换属性j的粒度层lj为lj-1
            % 计算最小的条件信息熵和对应的单粒度形式背景，最小条件信息熵多个时选择第一个最小。
            if j==1
                K(:,j)=Kj(:,j);
                tmpLH=LH(D,(K'),label,U_k0);
            else
                if tmpLH > LH(D,(K'),label,U_k0)
                    K(:,j)=Kj(:,j);
                    tmpLH=LH(D,(K'),label,U_k0);
                end
            end
        end
        
        layer=layer-1;% 粒度层逐次细化
    end
    optimals=K;
end
end




%-------------------------------------------SubFunction------------------------------------------------------%
function result=IE3(D,data,label)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/13
% @Parameter Instruction：CIE:Conditional  Entropy of Information
% function：计算覆盖Δ的条件信息熵（论文中 Definition 6）
% Input: D:等价类（分类数据集的类别标签）；data：多粒度覆盖粗糙决策表某个尺度的子表K = (l1,l2,· · · ,lm)（单粒度的形式背景）
% Output: result：覆盖Δ的条件信息熵H({d}|Δ) 或信息熵H(Δ)

U_k0=length(POS(D,data));%U_k = Pos_(Δ^k) ({d}),即Multi-scale covering rough sets with applications to data classification 一文中的决策d的正域
%% Codes
if label==0% CIE:Conditional  Entropy of Information
    C=cell(1,size(data,2));
    for i=1:size(data,2)% 由类属性块生成覆盖粗造集K
        C{i}=find(data(:,i)==1);
    end
    
    sums=0;
    for x=1:size(data,1)
        t=log2(size(intersect(CK(x,C),find(D==D(x))),1) /size(CK(x,C),1));
        if isempty(t)
            sums=sums+t;% log2(|Δx ∩ [x]d|/ |Δx|)
        end
    end
    result=-sums/size(data,1);
else% IE:  Entropy of Information
    C=cell(1,size(data,2));
    for i=1:size(data,2)% 由类属性块生成覆盖粗造集K
        C{i}=find(data(:,i)==1);
    end
    
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(CK(x,C),1) / size(data,1));% log2(|Δx |/ |U|)
    end
    result=-sums/size(data,1);
end

end


function result=CK(x,C)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/13
% @Parameter Instruction：
% reference:[1] Huang Z ,  Li J . Multi-scale covering rough sets with applications to data classification[J]. Applied Soft Computing, 2021(110-).
% function：For each x ∈ U, let the scaled covering neighborhood (x)_{C^K}（论文中Definition6）
% Input: C: 多尺度覆盖决策表某层的覆盖C（C is cell(1*n)，C{i}=find(data(:,i)==1),data:单粒度形式背景); x表示对象的下标值
% Output: result： (x)_{C^K}

counts=1;
tmp1=cell(1,1);
for i=1:size(C,2)% 计算包含x且是覆盖C的子集X
    if ~isempty(find(C{:,i}==x))
        tmp1{counts}=C{:,i};
        counts=counts+1;
    end
end

for i=1:size(tmp1,2)% 对所有的子集X求交
    if i==1
        result=tmp1{i};
    else
        result=intersect(tmp1{i},result);
    end
end
end


function result=LH(D,data,label,U_k0)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/13
% @Parameter Instruction：CIE:Conditional  Entropy of Information
% [2] DONGXIAO CHEN , Li J , Lin R , et al. Information Entropy and Optimal Scale Combination in Multi-Scale Covering Decision Systems[J]. IEEE Access, 2020, 8:182908-182917.
% 论文[2]中 Definition 14，与function result=IE(D,data,label) 类似。

%% Codes
% U_k0=length(POS(D,data));%U_k = Pos_(Δ^k) ({d}),即Multi-scale covering rough sets with applications to data classification 一文中的决策d的正域

if label==0% CIE:Conditional  Entropy of Information
    C=cell(1,size(data,2));
    for i=1:size(data,2)% 由类属性块生成覆盖粗造集K
        C{i}=find(data(:,i)==1);
    end
    
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(intersect(CK(x,C),find(D==D(x))),1) /U_k0);% log2(|Δx ∩ [x]d|/ |Δx|)
    end
    result=-sums/U_k0;
else% IE:  Entropy of Information
    C=cell(1,size(data,2));
    for i=1:size(data,2)% 由类属性块生成覆盖粗造集K
        C{i}=find(data(:,i)==1);
    end
    
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(CK(x,C),1) / U_k0);% log2(|Δx |/ |U|)
    end
    result=-sums/U_k0;
end

end


function result=POS(D,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/28
% @Parameter Instruction：
% function：计算决策的D的正域
% Input: D:等价类（分类数据集的类别标签）；data：单粒度覆盖粗糙决策表
% Output: result：决策的D的正域（对象的下标组成的集合）

counts=1;
tmp1=cell(1,1);
for i=unique(D')% 把标签D转换成等价类tmp1（文章中page4（3）（4））
   tmp1{counts,1}=find(D'==i);
   counts=counts+1;
end

result=[];
for i=1:size(tmp1,1)% 计算正域（文章中page4（5））
    result=[result ScaledApproximation(tmp1{i},data)];
end
result=unique(result);
end


function [SAL,SAU]=ScaledApproximation(X,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/28
% @Parameter Instruction：
% function：计算X关于覆盖C^K的覆盖上近似和覆盖下近似
% Input: X:对象子集的序号；data：单粒度覆盖粗糙决策表
% Output: [LTU,LTL]：覆盖上近似和覆盖下近似

C=cell(1,size(data,2));
for i=1:size(data,2)% 由类属性块生成覆盖粗造集K
    C{i}=find(data(:,i)==1);
end
   
SAL=[];
for i=1:size(data,1)% 计算下近似
    if SubSet(CK(i,C),X)==1 
        SAL=[SAL i];
    end
end

SAU=[];
for i=1:size(data,1)% 计算上近似
    if ~isempty(intersect(CK(i,C),X))
        SAU=[SAU i];
    end
end
end


function y=SubSet(A,B)% 判断A是不是B的子集（A，B是集合向量）
y=1;
for i=1:size(A)
    if ~isempty(find(B==A(i)))
        continue
    else
        y=0;
        break
    end
end
end