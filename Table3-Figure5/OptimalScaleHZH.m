function optimal=OptimalScaleHZH(tmpdata,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/31
% @Parameter Instruction： 

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
        
        if size(POS(D,(tmp1')),2)/AttrbuiteNum==1
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
        
        if size(POS(D,(tmp1')),2)/AttrbuiteNum==size(POS(D,(tmp1')),2)/AttrbuiteNum
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

function result=POS(D,data)
counts=1;
tmp1=cell(1,1);
for i=unique(D')
   tmp1{counts,1}=find(D'==i);
   counts=counts+1;
end

result=[];
for i=1:size(tmp1,1)
    result=[result ScaledApproximation(tmp1{i},data)];
end
result=unique(result);
end

function [SAL,SAU]=ScaledApproximation(X,data)
C=cell(1,size(data,2));
for i=1:size(data,2)
    C{i}=find(data(:,i)==1);
end
   
SAL=[];
for i=1:size(data,1)
    if SubSet(CK(i,C),X)==1 
        SAL=[SAL i];
    end
end

SAU=[];
for i=1:size(data,1)
    if ~isempty(intersect(CK(i,C),X))
        SAU=[SAU i];
    end
end
end

function y=SubSet(A,B)
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
counts=1;
tmp1=cell(1,1);
for i=1:size(C,2)
    if ~isempty(find(C{:,i}==x))
        tmp1{counts}=C{:,i};
        counts=counts+1;
    end
end

for i=1:size(tmp1,2)
    if i==1
        result=tmp1{i};
    else
        result=intersect(tmp1{i},result);
    end
end
end