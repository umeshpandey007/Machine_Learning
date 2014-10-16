function [ perr_per_run ] = monte_carlo_pla( fx,gx )
%monte_carlo_pla Monte Carlo simulaton for PLA

%Generate 500 data points between 1 and -1
num_points = 500;

tmp = ones(1,num_points);
datatmp = -1 + 2* rand(2,num_points);
Data = [tmp;datatmp];

fx_class = fx' * Data;

fx_class_final = (fx_class > 0);

gx_class = gx' * Data;

gx_class_final = (gx_class > 0);

result = fx_class_final+gx_class_final;

miscl_points = find(result == 1);

d = size(miscl_points);

perr_per_run = d(1,2) /num_points;


end

