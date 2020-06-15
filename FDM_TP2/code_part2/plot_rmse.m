clear
close all
clc


%% data size
s1 = 3883;
s2 = 6040;
s3 = 10;

%% Collect the rmse computations

outpath = './rmse';

rmse = [];
files = dir('./rmse/*.txt');

for i = 1:length(files)
    tmpRmse = load(['./rmse/' files(i).name]);
    rmse = [rmse;tmpRmse(:)];
end

plot(rmse, 'LineWidth',1.5);
title('rank = 80  step-size = 0.0001', 'FontSize',15);
set(gca,'linewidth',1.5,'FontSize',15)

ylabel('RMSE')
xlabel('Iteration');



