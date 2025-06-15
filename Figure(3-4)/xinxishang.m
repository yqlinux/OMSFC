function y=xinxishang(M)

%%
M_row=size(M,1);
xinxishang=0;
for i=1:M_row
    jiaoji=ones(1,M_row)';
    A=find(M(i,:)==1);
    for j=1:length(A)
        jiaoji=jiaoji&M(:,A(1,j));
    end
    d=sum(jiaoji);
    xinxishang=xinxishang+(1/(M_row))*((M_row-d)/(M_row));
end
y=xinxishang;

end
