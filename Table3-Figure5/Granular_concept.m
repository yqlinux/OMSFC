function y=Granular_concept(X)

%%
[m0,n0]=size(X);
concept = [];
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
