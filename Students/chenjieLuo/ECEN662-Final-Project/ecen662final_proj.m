clear;clc;
%% This part is used to collect and arrange data
load('data_725307741.mat');

%Collect data and remove the days containing Missing data
training_set = csvread('Train_data.csv');
re_arrange_index = find(training_set(:,4) == 0);
training_set(re_arrange_index,:) = [];

date = unique(training_set(:,3));
max_data_month = zeros(length(date),2);
max_data_month(:,1) = date;
for i = 1:length(date)
   max_data_month(i,2) = sum(training_set(:,3) == date(i));
end

date_max_station = max(max_data_month(:,2));
month_toCollect = max_data_month(max_data_month(:,2) == date_max_station,1);
date_size = length(month_toCollect);

training_data = zeros(date_max_station,date_size + 3);
training_data(:,1) = training_set(training_set(:,3)==month_toCollect(1),2);
for i = 1:1:length(month_toCollect)
    T = training_set(training_set(:,3) == month_toCollect(i),4);
    training_data(:,i+1) = T;         %Training_data contains the days which cover most complete data
end


stations = training_data(:,1);
for i = 1:length(stations)
    training_data(i,date_size+2:date_size+3) = data(data(:,1) == stations(i),33:34);
end

DATA = training_data(1:80,2:114)';
%% Start error detection
d = DATA(1:111,:);

[row,col] = size(d);
u = mean(d);

% This is my MLE test, after test MLE is of really small difference with
% cov, so we just use cov to make the computation easy.
U = ones(row,1) * u;
C1 = (d - U)' * (d - U)/30; 
C = cov(d);
position = [DATA(112,1:80)' DATA(113,1:80)'];

%% this part is the method to detect the single error

%Mapping lon and lat to x,y to compute distance, and to show the weather
%forecast stations on a map.
mstruct = defaultm('mercator');
mstruct.geoid=[6378,0.08];
mstruct.origin=[0,0,0];
mstruct = defaultm(mstruct);

[x,y] = projfwd(mstruct,position(:,1),position(:,2));

distance = pdist([x,y],'euclidean');
distance_mat = zeros(col,col);
counter = 1;
for j = 1:1:col-1
    for i = j+1:1:col
        distance_mat(i,j) = distance(counter);
        counter = counter + 1;
    end
end
for i = 1:1:col
    for j = i:1:col
        distance_mat(i,j) = distance_mat(j,i);
    end
end

figure;
scatter(x,y,50,u,'fill');
title('Average temperature of the whole country(03/2015)');
colorbar('Yticklabel',{'10F','20F','30F','40F','50F','60F','70F','80F'});

%% Apply distance_mapping to the covariance matrix to make the covariance matrix make more sense and error detection
V = randn(1,col);
Noise_mat = V'*V;
%[location_x, location_y] = find(ismember(distance_mat, max(distance_mat(:,:))));
%C = C + Noise_mat./exp(distance_mat/10000);  %????????????????C+ NOise??unbiased
f = - log(mvnpdf(d,u,C));


%Compute Conditional expectation for error detection
E = zeros(size(d,1),size(d,2));
for i = 1:length(d)
    for j = 1:size(d,2)
        tmp1 = d(i,:);
        o = ones(2000,1);
        o_sum = (o*tmp1)';
        k = linspace(u(j) - 3*sqrt(C(j,j)), u(j) + 3*sqrt(C(j,j)), 2000);
        o_sum(j,:) = k;
        p_con = sum(mvnpdf(o_sum',u,C));
        E_con = sum(mvnpdf(o_sum',u,C).*k')/p_con;
        E(i,j) = E_con;
    end
end


%Compute LOO estimate
f_LOO = zeros(length(d),1);
f_e_LOO = zeros(size(d,1),size(d,2));
count = 0;
for i = 1:length(d)
    tmp = d;
    tmp(i,:) = [];
    u_LOO = mean(tmp);
    C_LOO = cov(tmp);
    f_LOO(i) = - log(mvnpdf(d(i,:),u_LOO,C_LOO));
    for j = 1:size(d,2)
        tmp1 = d(i,:);
        tmp1(j) = tmp1(j) + 20;
        [~,pos] = max(abs(tmp1-E(i,:)));
        if pos == j
            count = count +1;
        end
        f_e_LOO(i,j) = - log(mvnpdf(tmp1,u_LOO,C_LOO));
    end
end

rate = count/80/111
    

threshold = mean(f_LOO);



%sum(sum(error>threshold))/(31*150)     
%joint_pdf = mvnpdf(DATA,u,C);