clear;clc;

load('input.mat');
row=128;
colum=128;
siz=size(re_images);
dd =[];

for i=1:siz(3)
    img=re_images(:,:,i);
    img=imresize(img,[row,colum]);
    img=im2double(img);
    d = feature_doG(img); 
    d = reshape(d,[1,size(d,1)*size(d,2)]);
    d = double(d);
    [d, mu, sigma] = featureNormalize(d);
    dd=[dd;d];
end

% [U, S] = pca(dd);
load('pca_afterDoG.mat')
K = 100;
Z = projectData(dd, U, K);

ii=randi([1,siz(3)]);
input=re_sketches(:,:,ii);
figure; imshow(input);
input=imresize(input,[row,colum]);
input=im2double(input);
input_mat = feature_doG(input);
input_mat = reshape(input_mat,[1,size(input_mat,1)*size(input_mat,2)]);
input_mat = double(input_mat);
[input_mat, mu, sigma] = featureNormalize(input_mat);
z = projectData(input_mat, U, K);

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