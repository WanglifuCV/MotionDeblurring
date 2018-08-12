clear;
clc;
close all;

addpath('epllcode');
addpath('whyte_code');
addpath('Test Images');

%% 使用选项

isselect = 0; %false or true
issimulation = 0;


%% Parameters1

opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %% the iterations (按理论上说应该是5或者7)
opts.k_thresh = 20; %% （按理论上说应该是20）

opts.reg_psf = 5; % default : 5;
opts.reg_latent = 0.001; % default : 0.001 


%% Load Image

%% Simulation
% filename = 'lena.png'; % succeed issimulation = 1;

%% Real Image
%% Dataset1
% filename = '37.png';% succeed / saturation = 0; / opts.reg_latent = 0.00015; / opts.reg_psf = 3; 
% saturation = 0;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 1; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 5; %% （按理论上说应该是20）

%% Dataset2
% filename = '7.jpg'; % succeed / saturation = 1; / opts.reg_latent = 0.003; / opts.reg_psf = 3; 
% saturation = 1;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 1; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 5; %% （按理论上说应该是20）

% %% Dataset3
% filename = 'lyndsey.tif'; %  succeed / saturation = 1;
% saturation = 1;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 1; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 5; %% （按理论上说应该是20）

%% Dataset4
filename = 'fishes.jpg'; %  succeed / saturation = 0; saturation = 1; / opts.reg_latent = 0.0003; / opts.reg_psf = 5; 
opts.prescale = 1; %%downsampling
opts.xk_iter = 7; %% the iterations (按理论上说应该是5或者7)
opts.k_thresh = 20; %% （按理论上说应该是20）
opts.reg_latent = 0.005;
opts.reg_psf = 5; 
saturation = 1;

%% Dataset5
% filename = 'Catoon.png';  % succeed / saturation = 0; / opts.reg_latent = 0.005; / opts.reg_psf = 5; 
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 7; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 20; %% （按理论上说应该是20）
% opts.reg_latent = 0.005;
% opts.reg_psf = 5; 
% saturation = 0;

%% Dataset6
% filename = 'boat_input.png'; % Not very good / saturation = 1; / opts.reg_latent = 0.0001; / opts.reg_psf = 5; 
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 5; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 20; %% （按理论上说应该是20）
% opts.reg_psf = 5; % default : 5;
% opts.reg_latent = 0.001; % default : 0.001 

%% Dataset7 Not Very Good
% filename = 'pietro.tif'; %  succeed / saturation = 1;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 5; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 20; %% （按理论上说应该是20）
% opts.reg_psf = 5; % default : 5;
% opts.reg_latent = 0.001; % default : 0.001 

%% Dataset 9 
% filename = 'censer.png';% succeed / saturation = 0; / opts.reg_latent = 0.00015; / opts.reg_psf = 3; 
% saturation = 1;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 5; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 20; %% （按理论上说应该是20）

%% Dataset 10
% filename = 'rhinoceros.png';% succeed / saturation = 0; / opts.reg_latent = 0.00015; / opts.reg_psf = 3; 
% saturation = 1;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 5; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 20; %% （按理论上说应该是20）
% opts.kernel_size = 55;
% opts.gamma_correct = 1;

%% Dataset 11
% filename = 'rhinoceros2.png';% succeed / saturation = 0; / opts.reg_latent = 0.00015; / opts.reg_psf = 3; 
% saturation = 0;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 1; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 5; %% （按理论上说应该是20）
% opts.kernel_size = 121;
% opts.gamma_correct = 2.2;

%% Dataset 12 Not Very Good
% filename = 'DSC01068.JPG';% succeed / saturation = 0; / opts.reg_latent = 0.00015; / opts.reg_psf = 3; 
% saturation = 1;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 1; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 5; %% （按理论上说应该是20）
% opts.kernel_size = 121;
% opts.gamma_correct = 1;

%% Dataset 13
% filename = 'pavilion.png';% succeed / saturation = 0; / opts.reg_latent = 0.00015; / opts.reg_psf = 3; 
% saturation = 0;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 5; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 20; %% （按理论上说应该是20）

%% Dataset 14
% filename = 'Blurred_iphone.jpg';% succeed / saturation = 0; / opts.reg_latent = 0.00015; / opts.reg_psf = 3; 
% saturation = 1;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 5; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 20; %% （按理论上说应该是20）

%% Dataset 15
% filename = 'IMG_9885.jpg';% succeed / saturation = 0; / opts.reg_latent = 0.00015; / opts.reg_psf = 3; 
% saturation = 1;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 5; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 20; %% （按理论上说应该是20）


%% Dataset 8 Coded Exposures

% 公共参数
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 1; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 5; %% （按理论上说应该是20）
% 
% opts.reg_psf = 5; % default : 5;
% opts.reg_latent = 0.001; % default : 0.001 

% saturation = 0;
% lambda_pixel = 4e-3;
% lambda_grad = 4e-3;
% lambda_tv = 0.01; 
% lambda_l0 = 1e-4; 
% weight_ring = 1;
% opts.kernel_size = 121;
% opts.gamma_correct = 1;

% Data1

% filename = '4.bmp'; % 编码曝光的，好极了。

% Data2
% filename = '6.bmp';

% Parameters
% opts.kernel_size = 221;
% opts.prescale = 1; %%downsampling
% opts.xk_iter = 1; %% the iterations (按理论上说应该是5或者7)
% opts.k_thresh = 5; %% （按理论上说应该是20）
% 
% opts.reg_psf = 5; % default : 5;
% opts.reg_latent = 0.001; % default : 0.001 

%% 效果不好的和待测试的


% filename = 'dong.jpg';

% filename = 'book.jpg'; %  Not very good / saturation = 1;

% filename = 'car6.png'; % fail / 该算法不适合这种图片
% opts.reg_psf = 0.8e1; 
% opts.reg_latent = 1.1e-6; 


% filename = '0015_blur65.png';  % fail /

% filename = 'WhiteBoard.png'; % fail /
% filename = 'blurred_outliers.png'; % fail /


% filename = 'IMG_1079.JPG'; %  Not very good / saturation = 1; 我猜是因为噪声才会这样

% filename = 'IMG_1196.JPG'; %  没成功
% 不过可以继续调参数。很有收获。需要考虑以下几个因素:reg_latent;reg_psf; k_thresh;双边滤波器的参数
% opts.reg_latent = 1e-1;
% opts.reg_psf = 8e1; 
% saturation = 0;
% opts.k_thresh = 5;

%% Parameters2
% saturation = 1;
lambda_pixel = 4e-3;
lambda_grad = 4e-3;
lambda_tv = 0.01; 
lambda_l0 = 1e-4; 
weight_ring = 1;
opts.kernel_size = 121;
opts.gamma_correct = 1;
%% PSF
if issimulation
    load( 'psf2.mat' );
    psf = psf( 1:end-1, 1:end-1 );
    psf = psf/sum( psf(:) );    
%     psf = fspecial( 'motion', 10, 45 );
%       psf = fspecial( 'gaussian', [ 10 10 ], 3 );
end

%% Read an Image
y0 = im2double(imread(filename));

if issimulation
    yc = imfilter( y0, psf,'conv', 'same' );    
    figure(1);imshow(yc);title( 'Blurred' );
    figure(2);imshow(psf,[]);title( 'Kernel' );
else
    yc = y0;
    figure(1);imshow(yc);title( 'Blurred' );
end

% Choose an area to deblur
if isselect ==1
    figure, imshow(yc);
    %tips = msgbox('Please choose the area for deblurring:');
    fprintf('Please choose the area for deblurring:\n');
    h = imrect;
    position = wait(h);
    close;
    B_patch = imcrop(yc,position);
    yc = (B_patch);
end



if size(yc,3)==3
    yg = im2double(rgb2gray(yc));
else
    yg = im2double(yc);
end

tic;
[kernel, interim_latent] = FastMotionDeblur(yg, lambda_pixel, lambda_grad, opts);
toc;

figure(3);
imshow(kernel,[]);

% kernel( kernel< max(kernel(:))/10 ) = 0;
kernel = kernel / sum( kernel(:) );

figure(4);
if issimulation
    subplot(1,2,1);imshow(kernel,[]);title('Estimated');
    subplot(1,2,2);imshow(psf,[]);title('Original');
else
    imshow(kernel,[]);title('Estimated');
end

figure(5);
imshow( interim_latent,[] );title( 'Estimated Latent' );

%% Deconv

%% Method 0



%% Method I
% 
% noiseSD = 0.01;
% patchSize = 8;
% 
% ks = floor((size(kernel, 1) - 1)/2);
% 
% % yg = padarray(yg, [1 1]*ks, 'replicate', 'both');
% for a=1:4
%   yc = edgetaper(yc, kernel);
% end
% 
% noiseI = yc;
% 
% y0 = yc;
% 
% % load GMM model
% load GSModel_8x8_200_2M_noDC_zeromean.mat
% 
% % uncomment this line if you want the total cost calculated
% % LogLFunc = @(Z) GMMLogL(Z,GS); 
% 
% % initialize prior function handle
% excludeList = [];
% prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);
% 
% % comment this line if you want the total cost calculated
% LogLFunc = [];
% 
% cleanI = [];
% 
% % deblur
% tic
% for c = 1 : size( noiseI, 3 )
%     [cleanI( :, :, c ), psnr,~] = EPLLhalfQuadraticSplitDeblur(noiseI( :, : ,c ),64/noiseSD^2,kernel,patchSize,50*[1 2 4 8 16 32 64],1,prior,y0(:,:,c),LogLFunc);
% end
% toc
% 
% % output result
% 
% figure(6);
% imshow(im2uint8(cleanI)); title('Restored Image');

%% Method II

if ~saturation
    %% 1. TV-L2 denoising method
    Latent = ringing_artifacts_removal(yc, kernel, lambda_tv, lambda_l0, weight_ring);
else
    %% 2. Whyte's deconvolution method (For saturated images)
    Latent = whyte_deconv(yc, kernel);
end

figure(7);
imshow( im2uint8(Latent) );title( 'Estimated Latent' );

imwrite( im2uint8(Latent), [ 'Results\deblur_' filename  ] );
kernel_show = kernel / max(kernel(:));
imwrite(im2uint8(kernel_show), [ 'Results\Kernel_' filename  ])