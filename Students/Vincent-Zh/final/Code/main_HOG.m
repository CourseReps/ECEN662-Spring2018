clear;clc;close all;
load('input.mat');

siz=size(re_images);
beta=0.25;
temp=imresize(re_images(:,:,1),beta);
re_image=zeros(size(temp,1),size(temp,2),siz(3));

for i=1:siz(3)
    re_image(:,:,i)=imresize(re_images(:,:,i),beta);
end

siz2=size(re_image);

features=[];
for ii=1:siz2(3)
    featureVector = extractHOGFeatures(uint8(re_image(:,:,ii)));
    features = [features;featureVector];
end

[U, S] = pca(features);
% load('svd_hog.mat')
K = 100;
Z = projectData(features, U, K);

close all;
index_sketch = randi(siz(3));
input_sketch=re_sketches(:,:,index_sketch);
figure;
displayData(reshape(double(input_sketch),[1,siz(1)*siz(2)]),siz(2));

input_vec=imresize(input_sketch,beta);
feature_input = extractHOGFeatures(input_vec);
z = projectData(feature_input, U, K);

thres = Z-ones(siz(3),1)*z;
thres1 = thres*thres';
[out,idx1] = sort(diag(thres1));

figure;
subplot(3,3,1)
imshow(re_images(:,:,idx1(1)));
title('Subplot 1')

subplot(3,3,2)
imshow(re_images(:,:,idx1(2)));
title('Subplot 2')

subplot(3,3,3)
imshow(re_images(:,:,idx1(3)));
title('Subplot 3')

subplot(3,3,4)
imshow(re_images(:,:,idx1(4)));
title('Subplot 4')

subplot(3,3,5)
imshow(re_images(:,:,idx1(5)));
title('Subplot 5')

subplot(3,3,6)
imshow(re_images(:,:,idx1(6)));
title('Subplot 6')

subplot(3,3,7)
imshow(re_images(:,:,idx1(7)));
title('Subplot 7')

subplot(3,3,8)
imshow(re_images(:,:,idx1(8)));
title('Subplot 8')

subplot(3,3,9)
imshow(re_images(:,:,idx1(9)));
title('Subplot 9')
