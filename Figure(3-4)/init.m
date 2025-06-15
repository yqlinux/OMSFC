function y=init(data)

%%
[~,columns]=size(data);
y=cell(columns-1,1);
for j=1:columns
    y{j} = [min(data(:,j)) max(data(:,j))];
end

end


