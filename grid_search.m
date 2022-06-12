% Perform grid search to find the optimal parameters for lambda and k
% In order to define the weight vector.
clc;clear;close all;
%% Load data
pth = 'C:\Users\crist\Desktop\Masters\First year\Second semester\Inverse problems in imaging\project\';
datafile = [pth,'2dt_heart.mat'];
load(datafile); % N - N - Nt - N_patients 128 x 128 x 11 x375
addpath('libs');
N = size(imgs, 1);
Nt = size(imgs, 3);
n_patients = 1; 
%% Calculations
% Define and initialize parameters
nRMSE_grid_search = zeros(10);
count = 0;
%% generate masks
acceleration_R = 5;
mask = rand(N, 1, Nt) < 1 / acceleration_R;
mask(floor(end/2)-5:floor(end/2)+5, :, :) = 1; % sample center of k-space densely

for k_val=0.1:0.1:1 % Loop through all values of k values (weights)
for lambda=0.01:0.01:0.1 
count = count +1;
fprintf('Iteration %d / 100 \n', count);
nRMSE_all = 0; % Initialize nRMSE
%% Calculate average nRMSE for all patients & generate visualizations
for kk=1:n_patients % loop through all patients (only one patient selected for speed-up purposes)
    fprintf('Analyzing patient... %d\n', kk); 
    img = imgs(:,:,:,kk); % select patient
    
    % Obtain k-space
    noise_std = 0.01; 
    kspc = i2k(img, 1);
    kspc = kspc + (randn(size(kspc)) + 1j * randn(size(kspc)))* noise_std; % add noise
    kspc = kspc .* mask;
    
    
    % TV reconstruction
    w = [k_val, k_val, 1-k_val]; % gradient weights in x, y and t dimensions
    rho = 1;  % penalty parameter
    maxiter = 30;
    cg_iters = 5; % Iterations for X subproblem
    
    [rec_tv, itinfo] = recon_admm_TV_2Dt_CAP(kspc,mask, lambda, w, rho, maxiter, cg_iters,1);
    
    % Compute reconstruction error
    error = sqrt(mean((abs(rec_tv(:)) - abs(img(:))).^2));
    % Normalize the error with the range of the reconstructed image: y_max - y_min
    y_max = max(abs(rec_tv(:)));y_min = min(abs(rec_tv(:)));
    nRMSE = error/(y_max - y_min);
    
    % Add error 
    nRMSE_all = nRMSE_all +nRMSE; %Add this patient error to the total error
    diary()  
end
nRMSE_all= nRMSE_all/n_patients;
nRMSE_grid_search(uint8(k_val*10),uint8(lambda*100))=nRMSE_all;
fprintf('k value is %d and lambda value is %d \n', k_val, lambda);
fprintf('nRMSE_grid_search is %d \n', nRMSE_all);
end
end
datafile = [pth,'nRMSE_grid_search_new.mat'];
save(datafile,'nRMSE_grid_search')
%% Plot grid table
pth = 'C:\Users\crist\Desktop\Masters\First year\Second semester\Inverse problems in imaging\project\';
datafile = [pth,'nRMSE_grid_search_new.mat'];
load(datafile);
figure;imagesc(nRMSE_grid_search);
set(gcf,'color','w');
xticks('');
yticks('');