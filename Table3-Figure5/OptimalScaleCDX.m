function optimal=OptimalScaleCDX(tmpdata,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/13
% @Parameter Instruction：
% reference：[1] DONGXIAO CHEN , Li J , Lin R , et al. Information Entropy and Optimal Scale Combination in Multi-Scale Covering Decision Systems[J]. IEEE Access, 2020, 8:182908-182917.

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
if IE3(D,(NewData{2,1}'),1) == 0
    optimal = ConsitentOpt(D,NewData);
else
    optimal = InconsitentOpt(D,NewData);
end

end


%% SubFunctions
function optimals=ConsitentOpt(D,data)
label=0;
if IE3(D,(data{2,end}'),label) == 0
    optimals=data{2,end};
elseif IE3(D,(data{2,end}'),label) > 0
    K=data{2,end};
    layer=size(data,2);
    while IE3(D,cell2mat(K'),label) > 0 && layer-1>0
        Kj=data{2,layer-1};
        for j=1:size(K,1)
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
        layer=layer-1;
    end
    optimals=K;
else
    optimals=data{2,end};
end
end

function optimals=InconsitentOpt(D,data)
label=0;
if IE3(D,(data{2,end}'),label) <= 0
    optimals=data{2,end};
elseif IE3(D,(data{2,end}'),label) > 0
    K=data{2,end};
    layer=size(data,2);
    while LH(D,(K'),label,length(POS(D,K'))) > 0 && layer-1>0
        U_k0=length(POS(D,K'));
        Kj=data{2,layer-1};
        for j=1:size(K,2)
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
        layer=layer-1;
    end
    optimals=K;
end
end

function result=IE3(D,data,label)
U_k0=length(POS(D,data));
if label==0
    C=cell(1,size(data,2));
    for i=1:size(data,2)
        C{i}=find(data(:,i)==1);
    end
    sums=0;
    for x=1:size(data,1)
        t=log2(size(intersect(CK(x,C),find(D==D(x))),1) /size(CK(x,C),1));
        if isempty(t)
            sums=sums+t;
        end
    end
    result=-sums/size(data,1);
else
    C=cell(1,size(data,2));
    for i=1:size(data,2)
        C{i}=find(data(:,i)==1);
    end
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(CK(x,C),1) / size(data,1));
    end
    result=-sums/size(data,1);
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

function result=LH(D,data,label,U_k0)
if label==0
    C=cell(1,size(data,2));
    for i=1:size(data,2)
        C{i}=find(data(:,i)==1);
    end
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(intersect(CK(x,C),find(D==D(x))),1) /U_k0);
    end
    result=-sums/U_k0;
else
    C=cell(1,size(data,2));
    for i=1:size(data,2)
        C{i}=find(data(:,i)==1);
    end
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(CK(x,C),1) / U_k0);
    end
    result=-sums/U_k0;
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
