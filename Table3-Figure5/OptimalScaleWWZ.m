function optimal=OptimalScaleWWZ(tmpdata,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/12
% @Parameter Instruction： [1] Wei-Zhi Wu, Yee Leung. Optimal scale selection for multi-scale decision tables - ScienceDirect[J]. International Journal of Approximate Reasoning, 2013, 54( 8):1107-1129.

%%
MT0=tmpdata{2,1};
MT1=tmpdata{1,1};
for i=1:length(MT1)
    ISC(i)=length(MT1{i})-1;
end

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


%% SubFunctions
function [optimals,marks]=ConsitentOpt(optimal,mark,D,data)
AttrbuiteNum=size(optimal,1);
LayerNum=size(data,2);

flag=0;
for j=1:AttrbuiteNum
    layer=2:LayerNum;
    if mark(1)==j
        layer(find(layer<=mark(2)))=[];
    end
    
    tmp1=optimal;
    for  k=layer
        tmp2=data{2,k};
        tmp1(j,1)=tmp2(j,1);
        
        if JudgeConsistent(D,(tmp1'))
            optimals=tmp1;
            marks(2)=k;
            flag=1;
            break
        end
    end
    
    if  (j==AttrbuiteNum && k==LayerNum) ||  LayerNum==1
        optimals=optimal;
        marks=[];
        break
    elseif flag==1
        marks(1)=j;
    end
end
end

function [optimals,marks]=InconsitentOpt(optimal,mark,D,data)
AttrbuiteNum=size(optimal,1);
LayerNum=size(data,2);

flag=0;
for j=1:AttrbuiteNum
    layer=2:LayerNum;
    if mark(1)==j
        layer(find(layer<=mark(2)))=[];
    end
    
    tmp1=optimal;
    for  k=layer
        tmp2=data{2,k};
        tmp1(j,1)=tmp2(j,1);
        
        if isequal(JudgeLAC(D,(data{2,1}')),JudgeLAC(D,(tmp1')))
            optimals=tmp1;
            marks(2)=k;
            flag=1;
            break
        end
    end

    if  (j==AttrbuiteNum && k==LayerNum) ||  LayerNum==1
        optimals=optimal;
        marks=[];
        break
    elseif flag==1
        marks(1)=j;
    end
end
end

function result=JudgeLAC(D,data)
[rows,columns]=size(data);
classes=(unique(D))';

result=[];
tmp1=zeros(rows,length(classes));
for i=1:length(classes)
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
[rows,columns]=size(data);
classes=(unique(D))';

tmp1=zeros(rows,length(classes));
for i=1:length(classes)
    indexs=find(D==classes(i));
    tmp1(indexs,i)=1;
    indexs=[];
end

counts=columns;
for i=1:columns
    for j=1:size(tmp1,2)
        if SubSet(data(:,i),tmp1(:,j))
            counts=counts-1;
            break
        end
    end
end

if counts==0
    result=true;
else 
    result=false;
end
end