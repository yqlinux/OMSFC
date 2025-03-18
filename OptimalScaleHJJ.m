function [MT0]=OptimalScaleHJJ(tmpdata,data)
%%  算法1 

for k=1:size(tmpdata{1,1},1)% 遍历每棵粒度树
    for layer=(size(tmpdata,2)):-1:2% 从粒度树第二层开始遍历剪枝，使用下层节点替换当前节点（不同粒度层替换只考虑两层之间）
        up=tmpdata{1,layer}{k,1};% 上层粒度树
        down=tmpdata{1,layer-1}{k,1};% 下层粒度树
        MaxStandard1=SubIE(tmpdata{1,1}{k,1},[data(:,k) data(:,end)]);% 计算最细粒度时的信息增益值
        OptScale{k,1}=up;

        for num=2:size(up,2)% 按顺序依次增加个数（同一层粒度剪枝组合），
            tmp_index=find(down <= up(num));% 寻找粒度树下一层替换节点
            TmpBreakPoint{1}=sort(unique([up down(tmp_index)]));

            Standard1=SubIE(cell2mat(TmpBreakPoint),[data(:,k) data(:,end)]);
            if (MaxStandard1 < Standard1) % 粒度粗，信息熵最大为最优粒度层
                OptScale{k,1}=sort(unique([up down(tmp_index)]));
                break;
            end
        end

    end
end
MT0=B2C(OptScale,data);
MT0 = MT0(:, any(MT0 ~= 0));% 删除全为0的列

end

%-------------------------------------------SubFunction------------------------------------------------------%
% +++++++++++++++++++++++++20220716+++++++++++++++++++++++++++++++%
function result=SubIE(breakpoint,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/25
% @Parameter Instruction：
% data：原数据
% breakpoint：数据集的断点
% result : 返回前子背景信息增益

%% 根据断点计算对应的形式背景
[rows,columns]=size(data);

dataresult=[];
for j=1:columns-1
    tmppoints=breakpoint;
    for k=1:size(tmppoints,2)-1
        tmp=[];
        for i=1:rows
            if(tmppoints(1,k) <= data(i,j) && data(i,j) < tmppoints(1,k+1))
                tmp=[tmp;1];
            else
                tmp=[tmp;0];
            end
        end
        dataresult=[dataresult tmp];
    end
end

%% 计算当前子背景信息增益值
result=xinxishang(dataresult);
end


function y=xinxishang(Context)
ConditionGC = unique(Granular_concept(Context), 'rows', 'stable');% 计算当前类属性块对应的条件形式背景对应的对象粒概念

y= 0;
G = size(ConditionGC,2);
for i=1:size(ConditionGC,1)
    ObjCC=ConditionGC(i,:);% 条件粒概念外延
    Pc = length(find(ObjCC == 1)) / G;% 概率 |x^(Δ*Δ)| / |U|
    y = y + (1 - Pc) / G;
end

end


function y=g_suanzi(B,x)

[x_r,~]=size(x);
if sum(B)==0
    y=ones(1,x_r);
else
    p=find(B==1);%B拥有的属性(代表一个矩阵)
    k=length(p);
    g=[];
    G=zeros(1,x_r);
    for i=1:k
        L=x(:,p(1,i))';
        g=[g;L];
        for j=1:x_r
            if all(g(:,j)==1)
                G(1,j)=1;
            else
                G(1,j)=0;
            end
        end
    end
    y=G;
end
end


function y=Granular_concept(X)

[m,~]=size(X);
GarCon=[];
for i=1:m
    intent=X(i,:);
    extent=g_suanzi(intent,X);
    GarCon=[GarCon;extent];
end
y=GarCon;
end


function Datas=B2C(breakpoint,data)%% 计算与断点对应的形式背景
% data：原数据; breakpoint：数据集的断点;


%% 根据断点计算对应的形式背景
[rows,columns]=size(data);

Datas=[];
for k=1:columns-1
    tmppoints=breakpoint{1,1};
    dataresult=Breakpoint2Context(tmppoints, data, k);
  
    S = 0;
    for ii=1:size(dataresult,2)
        S = S + dataresult(:,ii)*ii;
    end
    Datas = [Datas S];
end

end


function Context = Breakpoint2Context(breakpoint, data, k)
% Breakpoint2Context: 计算与断点对应属性k的形式背景
% breakpoint: 数据集的断点
% data: 原数据
% k: 第k个属性

% 获取数据的行数和断点的数量
[rows, ~] = size(data);
num_breakpoints = size(breakpoint, 2) - 1;

Context = zeros(rows, num_breakpoints); % 为形式背景预分配空间

for j = 1:num_breakpoints
    % 获取当前断点的范围
    lower_bound = breakpoint(1, j);
    upper_bound = breakpoint(1, j + 1);

    tmp = (data(:, k) >= lower_bound) & (data(:, k) < upper_bound);
    % 计算当前属性的形式背景
    if num_breakpoints==j
        tmp=tmp | (data(:, k) == upper_bound);
    end

    % 将计算得到的形式背景添加到结果矩阵中
    Context(:, j) = tmp;
end
% Context = Context(:, any(Context ~= 0));% 删除全为0的列
end