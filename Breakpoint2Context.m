function Context = Breakpoint2Context(breakpoint, data, k)

%%
[rows, ~] = size(data);
num_breakpoints = size(breakpoint, 2) - 1;

Context = zeros(rows, num_breakpoints); 

for j = 1:num_breakpoints
    lower_bound = breakpoint(1, j);
    upper_bound = breakpoint(1, j + 1);

    tmp = (data(:, k) >= lower_bound) & (data(:, k) < upper_bound);
    if num_breakpoints==j
        tmp=tmp | (data(:, k) == upper_bound);
    end

    Context(:, j) = tmp;
end
Context = Context(:, any(Context ~= 0));

end
