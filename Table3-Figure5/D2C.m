function D = D2C(data)

%%
classes = unique(data(:, end));
D = zeros(size(data, 1), length(classes));

for j = 1:length(classes)
    classIndices = data(:, end) == classes(j);
    D(classIndices, j) = 1;
end

D(~any(D, 2), end) = 1; 

end