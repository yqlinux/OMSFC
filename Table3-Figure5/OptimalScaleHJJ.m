function [MT0]=OptimalScaleHJJ(tmpdata,data)

%%
for k=1:size(tmpdata{1,1},1)
    for layer=(size(tmpdata,2)):-1:2
        up=tmpdata{1,layer}{k,1};
        down=tmpdata{1,layer-1}{k,1};
        MaxStandard1=SubIE(tmpdata{1,1}{k,1},[data(:,k) data(:,end)]);
        OptScale{k,1}=up;

        for num=2:size(up,2)
            tmp_index=find(down <= up(num));
            TmpBreakPoint{1}=sort(unique([up down(tmp_index)]));

            Standard1=SubIE(cell2mat(TmpBreakPoint),[data(:,k) data(:,end)]);
            if (MaxStandard1 < Standard1)
                OptScale{k,1}=sort(unique([up down(tmp_index)]));
                break;
            end
        end
    end
end
MT0=B2C(OptScale,data);
MT0 = MT0(:, any(MT0 ~= 0));

end


%% SubFunctions
function result=SubIE(breakpoint,data)
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

result=xinxishang(dataresult);
end

function y=xinxishang(Context)
ConditionGC = unique(Granular_concept(Context), 'rows', 'stable');

y= 0;
G = size(ConditionGC,2);
for i=1:size(ConditionGC,1)
    ObjCC=ConditionGC(i,:);
    Pc = length(find(ObjCC == 1)) / G;
    y = y + (1 - Pc) / G;
end
end

function y=g_suanzi(B,x)
[x_r,~]=size(x);
if sum(B)==0
    y=ones(1,x_r);
else
    p=find(B==1);
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

function Datas=B2C(breakpoint,data)
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
[rows, ~] = size(data);
num_breakpoints = size(breakpoint, 2) - 1;

Context = zeros(rows, num_breakpoints);

for j = 1:num_breakpoints
    lower_bound = breakpoint(1, j);
    upper_bound = breakpoint(1, j + 1);

    tmp = (data(:, k) >= lower_bound) & (data(:, k) < upper_bound);
    if num_breakpoints==j
        tmp=tmp | (data(:, k) == upper_bound);
    end

    Context(:, j) = tmp;
end
end