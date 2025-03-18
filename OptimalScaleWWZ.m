function optimal=OptimalScaleWWZ(tmpdata,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/12
% @Parameter Instruction： [1] Wei-Zhi Wu, Yee Leung. Optimal scale selection for multi-scale decision tables - ScienceDirect[J]. International Journal of Approximate Reasoning, 2013, 54( 8):1107-1129.
% function：一致和不一致（下近似）多尺度决策表最优粒度选择算法


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
if JudgeConsistent(D,(NewData{2,1}'))
    optimal=NewData{2,1};
    mark=[0,0];
    while(1)
        [optimals,marks]=ConsitentOpt(optimal,mark,D,NewData);
        optimal=optimals;
        mark=marks;
        if isempty(mark)
            break
        end
    end
else
    optimal=NewData{2,1};
    mark=[0,0];
    while(1)
        [optimals,marks]=InconsitentOpt(optimal,mark,D,NewData);
        optimal=optimals;
        mark=marks;
        if isempty(mark)
            break
        end
    end
end

end


function [optimals,marks]=ConsitentOpt(optimal,mark,D,data)% 一致信息系统下最优粒度（EC算法的数据集基本都是不一致多粒度信息系统）
%%% Tips：整个过程是把细粒度下类属性块替换成粗粒度类属性块实现（代码只替换单个，没有考虑组合）

AttrbuiteNum=size(optimal,1);
LayerNum=size(data,2);

flag=0;
for j=1:AttrbuiteNum % 遍历最细粒度下每一个属性
    layer=2:LayerNum;
    if mark(1)==j
        layer(find(layer<=mark(2)))=[];
    end
    
    tmp1=optimal;
    for  k=layer % 遍历当前粒度层下一层开始的层
        tmp2=data{2,k};
        tmp1(j,1)=tmp2(j,1);
        
        if JudgeConsistent(D,(tmp1'))% 找到H是一致的，结束对当前备选最优粒度的选择。
            optimals=tmp1;
            marks(2)=k;
            flag=1;
            break
        end
    end
    
    if  (j==AttrbuiteNum && k==LayerNum) ||  LayerNum==1% 当前备选粒度是最优粒度
        optimals=optimal;
        marks=[];
        break
    elseif flag==1
        marks(1)=j;
    end
end

end

function [optimals,marks]=InconsitentOpt(optimal,mark,D,data)% 不一致信息系统下的下近似最优粒度选择算法
%%% Tips：整个过程是把细粒度下类属性块替换成粗粒度类属性块实现，Definition 11 （1）的下近似最优粒度，

AttrbuiteNum=size(optimal,1);
LayerNum=size(data,2);

flag=0;
for j=1:AttrbuiteNum % 遍历最细粒度下每一个属性
    layer=2:LayerNum;
    if mark(1)==j
        layer(find(layer<=mark(2)))=[];
    end
    
    tmp1=optimal;
    for  k=layer % 遍历当前粒度层下一层开始的层
        tmp2=data{2,k};
        tmp1(j,1)=tmp2(j,1);
        
        if isequal(JudgeLAC(D,(data{2,1}')),JudgeLAC(D,(tmp1')))% 判断是否满足下近似一致，当下近似一致时更新下近似最优粒度
            optimals=tmp1;
            marks(2)=k;
            flag=1;
            break
        end
    end


    if  (j==AttrbuiteNum && k==LayerNum) ||  LayerNum==1% 当前备选粒度是最优粒度
        optimals=optimal;
        marks=[];
        break
    elseif flag==1
        marks(1)=j;
    end

end

end




%-------------------------------------------SubFunction------------------------------------------------------%
function result=JudgeLAC(D,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/07
% @Paper：[1] Wei-Zhi Wu, Yee Leung. Optimal scale selection for multi-scale decision tables - ScienceDirect[J]. International Journal of Approximate Reasoning, 2013, 54( 8):1107-1129.
% function：下近似计算，论文中（32）式下面
% Input: D:等价类（分类数据集的类别标签）；data：a single-scale decision table
% Output: result：L_{C^k}(d)的值

%% Codes
[rows,columns]=size(data);
classes=(unique(D))';

result=[];
tmp1=zeros(rows,length(classes));
for i=1:length(classes)% 决策属性转换成形式背景
    indexs=find(D==classes(i));
    tmp1(indexs,i)=1;
    indexs=[];
    
    for j=1:columns
        if SubSet(data(:,j),tmp1(:,i))
            result=[result data(:,j)];
        end
    end
end

 result=find(sum(result,2)~=0);

end

function y=SubSet(A,B)
% 判断A是不是B的子集（A，B是包含0和1的列集合向量）

y=true;
for i=1:size(A,1)
    if A(i,1) <= B(i,1)
        continue
    else
        y=false;
        break
    end
end
end

function result=JudgeConsistent(D,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/07
% @Parameter Instruction：A new approach of optimal scale selection to multi-scale decision tables
% function：Algorithm 1: Judging whether a SSDT is consistent or not.
% Input: D:等价类（分类数据集的类别标签）；data：a single-scale decision table
% Output: result：False or True

%% Codes
% step1
[rows,columns]=size(data);
classes=(unique(D))';

tmp1=zeros(rows,length(classes));
for i=1:length(classes)% 决策属性转换成形式背景
    indexs=find(D==classes(i));
    tmp1(indexs,i)=1;
    indexs=[];
end

% step2
counts=columns;
for i=1:columns
    for j=1:size(tmp1,2)
        if SubSet(data(:,i),tmp1(:,j))
            counts=counts-1;
            break
        end
    end
end

% step3
if counts==0
    result=true;
else 
    result=false;
end

end
