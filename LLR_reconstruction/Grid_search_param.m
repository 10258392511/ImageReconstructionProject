close all;
clear all;
clc;

%% Model- and Learning-Based Inverse Problems in Imaging FS22
% Group project
% Group members: Zhexin Wu, Cristina Almagro-Pérez, Runpu Hao
% Code author: Runpu Hao
% Data modified: 10/06/2022

% Locally low-rank reconstruction
% grid search for optimal lambda (regularization strength)
% Code adapted from tutorial 5
%% read the data

addpath(genpath('libs'));
addpath(genpath('proximal_ops'));
load('train_imgs.mat');
N = size(imgs, 1);
Nt = size(imgs, 3);
size_img = size(imgs); % N - N - Nt -Np (size of slices - number of timepoints - number of patients)
Num_patient = size_img(4); % number of patients

%% Create unique coil matrix
coils = ones(N, N, 1, 1);

%% generate undersampling masks

acceleration_R = 5;
mask = rand(N, 1, Nt) < 1 / acceleration_R;
mask(floor(end/2)-5:floor(end/2)+5, :, :) = 1;
% sample center of k-space densely (keep central 10 scanning lines)

%% iterate over lambda parameters
lambda_search = [0.01, 0.05, 0.1, 0.3, 0.5];

nRMSE_score = zeros(size(lambda_search));

for i_lambda = 1:numel(lambda_search)
    fprintf('-------------------------------')
    i_lambda
    fprintf('-------------------------------')
    % iterate over patients
    nRMSE = 0;
    for num_p = 1: Num_patient
        img_p = imgs(:,:,:,num_p);
        %% simulate acquisition, add reconstruction noise
        noise_std = 0.01;
        kspc = i2k(img_p .* coils, 1);
        kspc = kspc + noise_std * ( randn(size(kspc)) + 1j * randn(size(kspc)));
        kspc = kspc .* mask;
        %% zero-filled reconstruction
        rec_zf = sum(conj(coils) .* k2i(kspc, 1), 4) ./ sum(abs(coils).^2, 4);
        %% locally low rank reconstruction
        rec_pgd = pgd_nuclear(kspc, coils, mask, lambda_search(i_lambda), 12, 100);
        %% compute nRMSE
        nRMSE = nRMSE + nRMSE_comp(rec_pgd, img_p);
        num_p
    end
    nRMSE_score(i_lambda) = nRMSE / Num_patient;
end


plot(lambda_search, nRMSE_score, '*--');
set(gca,'XTick', (0:0.05:0.6))
xlabel('\lambda')
ylabel('nRMSE score')
title('Selection of optimal \lambda value')


