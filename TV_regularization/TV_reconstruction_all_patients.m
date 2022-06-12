% Code to generate the visualization of the TV-reconstructed images and
% compare them to the ground truth. We will also calculate the mean nRMSE
% for all patients.
clc;clear;close all;
%% Load data
pth = 'C:\Users\crist\Desktop\Masters\First year\Second semester\Inverse problems in imaging\project\';
datafile = 'test_imgs.mat'; % N - N - Nt - N_patients 128 x 128 x 11 x38
load(datafile);
addpath('libs');
N = size(imgs, 1);
Nt = size(imgs, 3);
n_patients = size(imgs,4);
%% Calculations
% Define and initialize parameters
undersampling_rates = 5; % [5, 10, 15, 20, 25];
noise_std = [0.01,0.02,0.03,0.04,0.05];
%nRMSE_all = zeros(1,length(undersampling_rates));
nRMSE_all = zeros(1,length(noise_std));

for i=1:length(undersampling_rates)
%% generate masks
fprintf('===> Analyzing with undersampling rate: %d\n', undersampling_rates(i)); 
acceleration_R = undersampling_rates(i); %3
mask = rand(N, 1, Nt) < 1 / acceleration_R;
mask(floor(end/2)-5:floor(end/2)+5, :, :) = 1; % sample center of k-space densely
%%
for j=1:length(noise_std)
    fprintf('===> Analyzing with noise std: %d\n', noise_std(j)); 
%% Calculate average nRMSE for all patients & generate visualizations
for kk=1:n_patients % loop through all patients
    fprintf('Analyzing patient... %d\n', kk); 
    img = imgs(:,:,:,kk); % select patient
    
    % Obtain k-space
    kspc = i2k(img, 1);
    kspc = kspc + (randn(size(kspc)) + 1j * randn(size(kspc)))* noise_std(j); % add noise
    kspc = kspc .* mask;
    

    % TV reconstruction
    w = [0.1, 0.1, 1]; % gradient weights in x, y and t dimensions
    rho = 1;  % penalty parameter
    maxiter = 30;
    cg_iters = 5; % Iterations for X subproblem
    [rec_tv, itinfo] = recon_admm_TV_2Dt_CAP(kspc,mask, 0.02, w, rho, maxiter, cg_iters,1);
    
    % Compute reconstruction error
    error = sqrt(mean((abs(rec_tv(:)) - abs(img(:))).^2));
    % Normalize the error with the range of the reconstructed image: y_max - y_min
    y_max = max(abs(rec_tv(:)));y_min = min(abs(rec_tv(:)));
    nRMSE = error/(y_max - y_min);
    
    
    % Add error 
    nRMSE_all(j) = nRMSE_all(j) + nRMSE;
    % Uncomment for visualization
%     if (kk == 1)%|| (kk == 2) || (kk==3) || (kk==4) || (kk==5) || (kk==6)
%         figure(kk);
%         subplot(2,2,1); imshow(abs(img(:,:, kk))); title('Original image');
%         subplot(2,2,2); imshow(abs(rec_tv(:,:, kk)));title('TV - reconstructed image');
%         subplot(2,2,3); imagesc(abs(img(:,:, kk))); title('Original image');daspect([1,1,1]);xticks([]);yticks([]);
%         subplot(2,2,4); imagesc(abs(rec_tv(:,:, kk)));title('TV - reconstructed image');
%         box on;
%         xticks([]);
%         yticks([]);
%         daspect([1,1,1]);
%         %sgtitle("Acceleration rate:" + num2str(acceleration_R));
%         sgtitle("Noise standard deviation:" + num2str(noise_std(j)));
%         %filename = strcat('patient_', num2str(kk), '_acc_rate_', num2str(acceleration_R), '.fig') ;
%         filename = strcat('patient_', num2str(kk), '_noise_std_', num2str(noise_std(j)), '.fig') ;
%         datafile = [pth,filename];
%         saveas(gcf,datafile)
%    end
    diary()
    
end
nRMSE_all(j) = nRMSE_all(j)/n_patients;
fprintf('nRMSE for acc_rate %d is %.5f \n', acceleration_R,nRMSE_all(j));
fprintf('nRMSE for noise_std %d is %.5f \n', noise_std(j),nRMSE_all(j)); 
end
end
datafile = [pth,'nRMSE_noise_std.mat'];
save(datafile,'nRMSE_all')
