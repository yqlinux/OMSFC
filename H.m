function y=H(attribute,A)
[m0,n0]=size(A);
attribute=find(attribute==1);
object=zeros(1,m0);
part_A=A(:,attribute);
for i=1:m0
    if all(part_A(i,:)==1)
        object(i)=1;
    end
end
y=object;
end
