function y=tiaojian(A,D)

%%
[m0,~]=size(A);
[~,~]=size(D);
X=eye(m0);
CGE0=[];
for i=1:m0
    B=[];
    X(i,:);
    a=H(L(X(i,:),A),A);
    a_old=a;
    b=H(L(X(i,:),D),D);
    for i=1:min(length(a),length(b))
        if a(i)==b(i) && a(i)==1 && b(i)==1
            B=[B a(i)];
        else
            a(i)=0;
            B=[B a(i)];
        end
    end
    B;
    sum1=(sum(a_old)-sum(B))/m0^2;
    CGE0=[CGE0,sum1];
end
CGE=sum(CGE0);
y=CGE;

end
