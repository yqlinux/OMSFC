function optimal=OptimalScaleHZH(tmpdata,data)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/31
% @Parameter Instruction�� 
% function��һ�ºͲ�һ�¶�߶Ⱦ��߱���������ѡ���㷨


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


function [optimals,marks]=ConsitentOpt(optimal,mark,D,data)% һ����Ϣϵͳ�����������ж�
%%% Tips�����������ǰ�ϸ�����������Կ��滻�ɴ����������Կ�ʵ�֣�����8��Hѡ����NP���⣨����ֻ�滻������û�п�����ϣ�

AttrbuiteNum=size(optimal,1);
LayerNum=size(data,2);

flag=0;
for j=1:AttrbuiteNum % ������ϸ������ÿһ������
    layer=2:LayerNum;
    if mark(1)==j
        layer(find(layer<=mark(2)))=[];
    end
    
    tmp1=optimal;
    for  k=layer % ������ǰ���Ȳ���һ�㿪ʼ�Ĳ�
        tmp2=data{2,k};
        tmp1(j,1)=tmp2(j,1);
        
        if size(POS(D,(tmp1')),2)/AttrbuiteNum==1% �ҵ�H��һ�µģ������Ե�ǰ��ѡ�������ȵ�ѡ�񡣣������� Definition 8 �� (2)��
            optimals=tmp1;
            marks(2)=k;
            flag=1;
            break
        end
    end
    
    if  (j==AttrbuiteNum && k==LayerNum) ||  LayerNum==1% ��ǰ��ѡ��������������
        optimals=optimal;
        marks=[];
        break
    elseif flag==1
        marks(1)=j;
    end
end

end


function [optimals,marks]=InconsitentOpt(optimal,mark,D,data)% ��һ����Ϣϵͳ�����������ж�
%%% Tips�����������ǰ�ϸ�����������Կ��滻�ɴ����������Կ�ʵ�֣�����8��Hѡ����NP���⣨����ֻ�滻������û�п�����ϣ�

AttrbuiteNum=size(optimal,1);
LayerNum=size(data,2);

flag=0;
for j=1:AttrbuiteNum % ������ϸ������ÿһ������
    layer=2:LayerNum;
    if mark(1)==j
        layer(find(layer<=mark(2)))=[];
    end
    
    tmp1=optimal;
    for  k=layer % ������ǰ���Ȳ���һ�㿪ʼ�Ĳ�
        tmp2=data{2,k};
        tmp1(j,1)=tmp2(j,1);
        
        if size(POS(D,(tmp1')),2)/AttrbuiteNum==size(POS(D,(tmp1')),2)/AttrbuiteNum % �ҵ�H ������������ Definition9��2��
            optimals=tmp1;
            marks(2)=k;
            flag=1;
            break
        end
    end
    
    if  (j==AttrbuiteNum && k==LayerNum) ||  LayerNum==1% ��ǰ��ѡ��������������
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

function result=CK(x,C)
% @Author: Chen QiangQiang(2434302354@qq.com);
% @Date  : 2022/08/28
% @Parameter Instruction��
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