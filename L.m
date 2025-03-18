function y=L(object,A)
[m0,n0]=size(A);
object=find(object==1);
attribute=zeros(1,n0);
part_A=A(object,:);
for i=1:n0
    if all(part_A(:,i)==1)
        attribute(i)=1;
    end
end
y=attribute;
end
        
    