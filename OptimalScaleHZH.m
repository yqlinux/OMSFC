function optimal=OptimalScaleHZH(tmpdata,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/31
% @Parameter Instruction： 
% function：一致和不一致多尺度决策表最优粒度选择算法


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
        MT1(:,jj+First_Ind(ii))=MT1(:,jj-1+First_Ind(ii))+MT0(:,jj+First_Ind(ii))*jj;
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

NewData = tmpdata;
if length(POS(D,(NewData{2,1}')))==size((NewData{2,1}'),1)
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


function [optimals,marks]=ConsitentOpt(optimal,mark,D,data)% 一致信息系统下最优粒度判断
%%% Tips：整个过程是把细粒度下类属性块替换成粗粒度类属性块实现，定义8中H选择是NP问题（代码只替换单个，没有考虑组合）

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
        
        if size(POS(D,(tmp1')),2)/AttrbuiteNum==1% 找到H是一致的，结束对当前备选最优粒度的选择。（论文中 Definition 8 中 (2)）
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


function [optimals,marks]=InconsitentOpt(optimal,mark,D,data)% 不一致信息系统下最优粒度判断
%%% Tips：整个过程是把细粒度下类属性块替换成粗粒度类属性块实现，定义8中H选择是NP问题（代码只替换单个，没有考虑组合）

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
        
        if size(POS(D,(tmp1')),2)/AttrbuiteNum==size(POS(D,(tmp1')),2)/AttrbuiteNum % 找到H 不满足论文中 Definition9（2）
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

function result=CK(x,C)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/28
% @Parameter Instruction：
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