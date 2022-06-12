close all;
clear all;
clc;

%% Model- and Learning-Based Inverse Problems in Imaging FS22
% Group project
% Group members: Zhexin Wu, Cristina Almagro-Pérez, Runpu Hao
% Code author: Runpu Hao
% Data modified: 10/06/2022

% Locally low-rank reconstruction
% Test for optimal number of iterations
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

%% locally low rank reconstruction
itera = [5, 10, 20, 50, 100, 200, 300, 400];
nRMSE_score=zeros(size(itera));
for i_iter = 1:numel(itera)
    fprintf('-------------------------------')
    i_iter
    fprintf('-------------------------------')
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
        rec_pgd = pgd_nuclear(kspc, coils, mask,  0.5, 12, itera(i_iter));
        nRMSE = nRMSE + nRMSE_comp(rec_pgd, img_p);
        num_p
    end
    nRMSE_score(i_iter) = nRMSE/Num_patient;
end
%% compute nRMSE
plot(itera, nRMSE_score, '*-');
title('optimal iterations for convergence')
xlabel('Number of iterations')
ylabel('nRMSE score')