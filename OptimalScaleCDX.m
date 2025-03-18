function optimal=OptimalScaleCDX(tmpdata,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/13
% @Parameter Instruction��
% reference��[1] DONGXIAO CHEN , Li J , Lin R , et al. Information Entropy and Optimal Scale Combination in Multi-Scale Covering Decision Systems[J]. IEEE Access, 2020, 8:182908-182917.


%% ��������ʽ��������߶Ȼ�Ϊ��߶���Ϣϵͳ
MT0=tmpdata{2,1};
MT1=tmpdata{1,1};
for i=1:length(MT1)
    ISC(i)=length(MT1{i})-1;
end

% ����߶Ȼ�����ʽ��������Ķ����ȱ�����ۡ�
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
label=0; % IE3����Ķ���������Ϣ��
if IE3(D,(data{2,end}'),label) == 0
    optimals=data{2,end};
elseif IE3(D,(data{2,end}'),label) > 0
    %% ʹ��������Ⱥ͹�����������㷨���Ӷ���ͬ����ͬ�����ݼ���Ч�ʿ��ܲ���ͬ���Ľ������㷨Ϊ��Ҷ˹�������������Ч�ʡ�
    % step4-> step6
    K=data{2,end};% ��ʼ��KΪ������ȶ�������ʽ����
    layer=size(data,2);% ������Ȳ����
    while IE3(D,cell2mat(K'),label) > 0 && layer-1>0 % H({d}|��^K) > 0
        Kj=data{2,layer-1};
        for j=1:size(K,1)% �滻����j�����Ȳ�ljΪlj-1
            % ������С��������Ϣ�غͶ�Ӧ�ĵ�������ʽ��������С������Ϣ�ض��ʱѡ���һ����С��
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
        
        layer=layer-1;% ���Ȳ����ϸ��
    end
    optimals=K;
else
    optimals=data{2,end};
end
end

function optimals=InconsitentOpt(D,data)%Algorithm 2 The Optimal Scale Combination of a Inconsistent Multi-Scale Covering Decision System
% step3
label=0; % IE3����Ķ���������Ϣ��
if IE3(D,(data{2,end}'),label) <= 0
    optimals=data{2,end};
elseif IE3(D,(data{2,end}'),label) > 0
    %% ʹ��������Ⱥ͹�����������㷨���Ӷ���ͬ����ͬ�����ݼ���Ч�ʿ��ܲ���ͬ���Ľ������㷨Ϊ��Ҷ˹�������������Ч�ʡ�
    % step4-> step6
    K=data{2,end};% ��ʼ��KΪ������ȶ�������ʽ����
    layer=size(data,2);% ������Ȳ����
    while LH(D,(K'),label,length(POS(D,K'))) > 0 && layer-1>0 % H({d}|��^K) > 0�����������˼·��
        U_k0=length(POS(D,K'));%U_k = Pos_(��^k) ({d}),��Multi-scale covering rough sets with applications to data classification һ���еľ���d������
        Kj=data{2,layer-1};
        for j=1:size(K,2)% �滻����j�����Ȳ�ljΪlj-1
            % ������С��������Ϣ�غͶ�Ӧ�ĵ�������ʽ��������С������Ϣ�ض��ʱѡ���һ����С��
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
        
        layer=layer-1;% ���Ȳ����ϸ��
    end
    optimals=K;
end
end




%-------------------------------------------SubFunction------------------------------------------------------%
function result=IE3(D,data,label)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/13
% @Parameter Instruction��CIE:Conditional  Entropy of Information
% function�����㸲�Ǧ���������Ϣ�أ������� Definition 6��
% Input: D:�ȼ��ࣨ�������ݼ�������ǩ����data�������ȸ��Ǵֲھ��߱�ĳ���߶ȵ��ӱ�K = (l1,l2,�� �� �� ,lm)�������ȵ���ʽ������
% Output: result�����Ǧ���������Ϣ��H({d}|��) ����Ϣ��H(��)

U_k0=length(POS(D,data));%U_k = Pos_(��^k) ({d}),��Multi-scale covering rough sets with applications to data classification һ���еľ���d������
%% Codes
if label==0% CIE:Conditional  Entropy of Information
    C=cell(1,size(data,2));
    for i=1:size(data,2)% �������Կ����ɸ��Ǵ��켯K
        C{i}=find(data(:,i)==1);
    end
    
    sums=0;
    for x=1:size(data,1)
        t=log2(size(intersect(CK(x,C),find(D==D(x))),1) /size(CK(x,C),1));
        if isempty(t)
            sums=sums+t;% log2(|��x �� [x]d|/ |��x|)
        end
    end
    result=-sums/size(data,1);
else% IE:  Entropy of Information
    C=cell(1,size(data,2));
    for i=1:size(data,2)% �������Կ����ɸ��Ǵ��켯K
        C{i}=find(data(:,i)==1);
    end
    
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(CK(x,C),1) / size(data,1));% log2(|��x |/ |U|)
    end
    result=-sums/size(data,1);
end

end


function result=CK(x,C)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/09/13
% @Parameter Instruction��
% reference:[1] Huang Z ,  Li J . Multi-scale covering rough sets with applications to data classification[J]. Applied Soft Computing, 2021(110-).
% function��For each x �� U, let the scaled covering neighborhood (x)_{C^K}��������Definition6��
% Input: C: ��߶ȸ��Ǿ��߱�ĳ��ĸ���C��C is cell(1*n)��C{i}=find(data(:,i)==1),data:��������ʽ����); x��ʾ������±�ֵ
% Output: result�� (x)_{C^K}

counts=1;
tmp1=cell(1,1);
for i=1:size(C,2)% �������x���Ǹ���C���Ӽ�X
    if ~isempty(find(C{:,i}==x))
        tmp1{counts}=C{:,i};
        counts=counts+1;
    end
end

for i=1:size(tmp1,2)% �����е��Ӽ�X��
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
% @Parameter Instruction��CIE:Conditional  Entropy of Information
% [2] DONGXIAO CHEN , Li J , Lin R , et al. Information Entropy and Optimal Scale Combination in Multi-Scale Covering Decision Systems[J]. IEEE Access, 2020, 8:182908-182917.
% ����[2]�� Definition 14����function result=IE(D,data,label) ���ơ�

%% Codes
% U_k0=length(POS(D,data));%U_k = Pos_(��^k) ({d}),��Multi-scale covering rough sets with applications to data classification һ���еľ���d������

if label==0% CIE:Conditional  Entropy of Information
    C=cell(1,size(data,2));
    for i=1:size(data,2)% �������Կ����ɸ��Ǵ��켯K
        C{i}=find(data(:,i)==1);
    end
    
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(intersect(CK(x,C),find(D==D(x))),1) /U_k0);% log2(|��x �� [x]d|/ |��x|)
    end
    result=-sums/U_k0;
else% IE:  Entropy of Information
    C=cell(1,size(data,2));
    for i=1:size(data,2)% �������Կ����ɸ��Ǵ��켯K
        C{i}=find(data(:,i)==1);
    end
    
    sums=0;
    for x=1:size(data,1)
        sums=sums+log2(size(CK(x,C),1) / U_k0);% log2(|��x |/ |U|)
    end
    result=-sums/U_k0;
end

end


function result=POS(D,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/28
% @Parameter Instruction��
% function��������ߵ�D������
% Input: D:�ȼ��ࣨ�������ݼ�������ǩ����data�������ȸ��Ǵֲھ��߱�
% Output: result�����ߵ�D�����򣨶�����±���ɵļ��ϣ�

counts=1;
tmp1=cell(1,1);
for i=unique(D')% �ѱ�ǩDת���ɵȼ���tmp1��������page4��3����4����
   tmp1{counts,1}=find(D'==i);
   counts=counts+1;
end

result=[];
for i=1:size(tmp1,1)% ��������������page4��5����
    result=[result ScaledApproximation(tmp1{i},data)];
end
result=unique(result);
end


function [SAL,SAU]=ScaledApproximation(X,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/28
% @Parameter Instruction��
% function������X���ڸ���C^K�ĸ����Ͻ��ƺ͸����½���
% Input: X:�����Ӽ�����ţ�data�������ȸ��Ǵֲھ��߱�
% Output: [LTU,LTL]�������Ͻ��ƺ͸����½���

C=cell(1,size(data,2));
for i=1:size(data,2)% �������Կ����ɸ��Ǵ��켯K
    C{i}=find(data(:,i)==1);
end
   
SAL=[];
for i=1:size(data,1)% �����½���
    if SubSet(CK(i,C),X)==1 
        SAL=[SAL i];
    end
end

SAU=[];
for i=1:size(data,1)% �����Ͻ���
    if ~isempty(intersect(CK(i,C),X))
        SAU=[SAU i];
    end
end
end


function y=SubSet(A,B)% �ж�A�ǲ���B���Ӽ���A��B�Ǽ���������
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