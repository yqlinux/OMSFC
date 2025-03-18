function [dataresult]=OptimalScaleZQH(tmpdata,data)

DATA=data; %调入的是原始的决策表
CON=DATA(:,[1:end-1]);
d0=DATA(:,end);
difv=unique(sort(d0));
stand=1:length(difv);
if all(difv==stand')
    D=d0;
else
    D=zeros(length(d0),1);
    for i=1:length(difv)
        loc=find(d0==difv(i));
        D(loc)=i;
    end
end
[N,M]=size(CON);

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
    
   
isc1=ones(1,M);
ind=[0 ISC(1:end-1)];
First_Ind=cumsum(ind);

%求一个最优尺度组合
My_OSC1=My_firstred(MT0,D,ISC,isc1,M,First_Ind,0);
dataresult=[];
for i=1:length(My_OSC1)
    dataresult=[dataresult MT0(:,My_OSC1(i)+First_Ind(i))];
end
end


%-------------------------------------------SubFunction------------------------------------------------------%
% +++++++++++++++++++++++++20220616+++++++++++++++++++++++++++++++%
function opt=My_firstred(MT0,D,ISC,isc,M,First_Ind,flag_mult)%直接从最细粒度ISC出发，求一个最优组合
mcolu=First_Ind+ISC;
MCON=MT0(:,mcolu);
[D_ISC,firstloc]=CT_ISC(MCON,D);
C_ISC=MT0(firstloc',:);
A1=D_ISC;
A1(A1~=0)=1;
loc_pos=find(sum(A1,2)==1);
Npos0=sum( sum( D_ISC(loc_pos',:),2 ) );
order=randperm(M);
[opt,~]=Mstepopt_PR(C_ISC,D_ISC,First_Ind,isc,Npos0,[],[],ISC,order);
end

function [CTD,firstind]=CT_ISC(CON,D)
[A,firstind,newind]=unique(CON,'rows');
V=[newind D];
CTD=accumarray(V,1);
end


function [opt,inconc]=Mstepopt_PR(oldC,oldD,First_Ind,isc,Npos0,OSC,INCONC,sk0,order)%从一个一致尺度尺度组合sk0出发，逐个考虑每一个属性的最优尺度
inconc=[];
opt=sk0;
if isempty(OSC)>0 & isempty(INCONC)>0
    for i=1:length(order)
        inconsk=[];
        opt(order(i))=isc(order(i));
        while opt(order(i))<sk0(order(i))
            [flag,oldC,oldD]=MConsist_PR(oldC,oldD,First_Ind,Npos0,opt);
            if flag==1
                break
            else
                inconsk=opt;
                opt(order(i))=opt(order(i))+1;
            end
        end
        inconc=[inconc;inconsk];
    end
elseif isempty(OSC)>0 & isempty(INCONC)==0
    for i=1:length(order)
        inconsk=[];
        opt(order(i))=isc(order(i));
        while opt(order(i))<sk0(order(i))
            if  any(all(bsxfun(@le,opt, INCONC),2))==1
                opt(order(i))=opt(order(i))+1;
            else
                [flag,oldC,oldD]=MConsist_PR(oldC,oldD,First_Ind,Npos0,opt);
                if flag==0
                    inconsk=opt;
                    opt(order(i))=opt(order(i))+1;
                else
                    break
                end
            end
        end
        inconc=[inconc;inconsk];
    end
    
elseif isempty(OSC)==0 & isempty(INCONC)>0
    for i=1:length(order)
        inconsk=[];
        opt(order(i))=isc(order(i));
        while opt(order(i))<sk0(order(i))
            if  any(all(bsxfun(@ge,opt, OSC),2))==1 | any(all(bsxfun(@le,opt, OSC),2))==1 % 与OSC可比较，opt不在边界域
                opt(order(i))=opt(order(i))+1;
            else
                [flag,oldC,oldD]=MConsist_PR(oldC,oldD,First_Ind,Npos0,opt);
                if flag==0
                    inconsk=opt;
                    opt(order(i))=opt(order(i))+1;
                else
                    break
                end
            end
        end
        inconc=[inconc;inconsk];
    end
else
    for i=1:length(order)
        inconsk=[];
        opt(order(i))=isc(order(i));
        while opt(order(i))<sk0(order(i))
            if  any(all(bsxfun(@ge,opt, OSC),2))==1 | any(all(bsxfun(@le,opt, OSC),2))==1 | any(all(bsxfun(@le,opt, INCONC),2))==1 % 与OSC可比较，或是INCONC的下界，opt不在边界域
                opt(order(i))=opt(order(i))+1;
            else
                [flag,oldC,oldD]=MConsist_PR(oldC,oldD,First_Ind,Npos0,opt);
                if flag==0
                    inconsk=opt;
                    opt(order(i))=opt(order(i))+1;
                else
                    break
                end
            end
        end
        inconc=[inconc;inconsk];
    end
end
if isempty(inconc)==0
    ind1=all(bsxfun(@ge,opt, inconc),2);
    ind2=find(ind1==0);
    inconc=inconc(ind2',:);
end
end

function [flag,newMC,newD]=MConsist_PR(oldMC,oldD,First_Ind,Npos0,sk)%一致性
loc1=find(sk~=0);
sk_loc=First_Ind(loc1)+sk(loc1);
sk_con=oldMC(:,sk_loc);
[~,firstind,newind]=unique(sk_con,'rows');
newMC=oldMC(firstind',:);
newD=[];
for j=1:size(oldD,2)
    newD=[newD accumarray(newind,oldD(:,j)')];
end
A=newD;
A(A~=0)=1;
loc2=find(sum(A,2)==1);
B=newD(loc2',:);
Npos=sum(sum(B,2));
if Npos==Npos0
    flag=1;
else
    flag=0;
    newMC=oldMC;
    newD=oldD;
end
end