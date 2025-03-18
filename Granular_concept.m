function y=Granular_concept(X)%对象粒概念

[m0,n0]=size(X);%m0:对象个数，n0：属性个数
concept=[];
%m0:对象数；n0：属性数
for i=1:m0
    x=[zeros(1,m0+n0)];
    A=[];
    for j=1:m0
    A=find(X(i,:)==1);
    x(A'+m0)=1;
    if all(X(j,A)==X(i,A))
       x(j)=1;
    end
   end
concept=[concept;x];
end

y=concept;
end
