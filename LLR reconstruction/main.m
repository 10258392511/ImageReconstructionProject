close all;
clear all;
clc;

%% Model- and Learning-Based Inverse Problems in Imaging FS22
% Group project
% Group members: Zhexin Wu, Cristina Almagro-Pérez, Runpu Hao
% Code author: Runpu Hao
% Data modified: 10/06/2022

% Locally low-rank reconstruction
% Computes nRMSE score under optimized hyperparameters, at different
% acceleration rates and different noise standard deviations separately
% Code adapted from tutorial 5
%% read the data

addpath(genpath('libs'));
addpath(genpath('proximal_ops'));
load('test_imgs.mat');
N = size(imgs, 1);
Nt = size(imgs, 3);
size_img = size(imgs); % N - N - Nt -Np (size of slices - number of timepoints - number of patients)
Num_patient = size_img(4); % number of patients

%% Create unique coil matrix
coils = ones(N, N, 1, 1);

%% Computation of nRMSE score on the test set with different acceleration rates and fixed noise std
% acceleration rate = [5, 10, 15, 20, 25]
% noise std is fixed at 0.01

acceleration_R = [5, 10, 15, 20, 25];

fprintf('Varying acceleration rate; fixed noise std = 0.01:\n')
for i_acc = 1:numel(acceleration_R)
    nRMSE_score = 0;
    for patient = 1:Num_patient
        %% generate undersampling masks
        mask = rand(N, 1, Nt) < 1 / acceleration_R(i_acc);
        mask(floor(end/2)-5:floor(end/2)+5, :, :) = 1;
        % sample center of k-space densely (keep central 10 scanning lines)

        %% Iterate over all patients in test set
        img_p = imgs(:,:,:,patient);

        %% simulate acquisition, add reconstruction noise
        noise_std = 0.01; % fixed noise std
        kspc = i2k(img_p .* coils, 1);
        kspc = kspc + noise_std * ( randn(size(kspc)) + 1j * randn(size(kspc)));
        kspc = kspc .* mask;
        %% zero-filled reconstruction
        rec_zf = sum(conj(coils) .* k2i(kspc, 1), 4) ./ sum(abs(coils).^2, 4);

        %% locally low rank reconstruction
        rec_pgd = pgd_nuclear(kspc, coils, mask,  0.3, 12, 100);

        %% compute nRMSE
        nRMSE_score = nRMSE_score + nRMSE_comp(rec_pgd, img_p);
    end
    nRMSE_score = nRMSE_score/Num_patient;
    fprintf(['Acceleration rate = ', num2str(acceleration_R(i_acc)), ', nRMSE score is: ', num2str(nRMSE_score), '\n'])
end

fprintf('----------------------------------------\n')

%% Computation of nRMSE score on the test set with different noise std and fixed acceleration rate
% noise std = [0.01, 0.02, 0.03, 0.04, 0.05]
% acceleration rate is fixed at 5

noise_std = [0.01, 0.02, 0.03, 0.04, 0.05];

fprintf('Varying noise std; fixed acceleration rate = 5:\n')
for i_noise = 1:numel(noise_std)
    nRMSE_score = 0;
    for patient_n = 1:Num_patient
        %% generate undersampling masks
        acceleration_R = 5; % fixed
        mask = rand(N, 1, Nt) < 1 / acceleration_R;
        mask(floor(end/2)-5:floor(end/2)+5, :, :) = 1;
        % sample center of k-space densely (keep central 10 scanning lines)

        %% Iterate over all patients in test set
        img_p = imgs(:,:,:,patient_n);

        %% simulate acquisition, add reconstruction noise
        noise_std_iter = noise_std(i_noise);
        kspc = i2k(img_p .* coils, 1);
        kspc = kspc + noise_std_iter * ( randn(size(kspc)) + 1j * randn(size(kspc)));
        kspc = kspc .* mask;
        %% zero-filled reconstruction
        rec_zf = sum(conj(coils) .* k2i(kspc, 1), 4) ./ sum(abs(coils).^2, 4);

        %% locally low rank reconstruction
        rec_pgd = pgd_nuclear(kspc, coils, mask,  0.3, 12, 100);

        %% compute nRMSE
        nRMSE_score = nRMSE_score + nRMSE_comp(rec_pgd, img_p);
    end
    nRMSE_score = nRMSE_score/Num_patient;
    fprintf(['Noise std = ', num2str(noise_std(i_noise)), ', nRMSE score is: ', num2str(nRMSE_score), '\n'])
end
fprintf('-------finished-------\n')
        
        
        
        
        
        
        
        
        
