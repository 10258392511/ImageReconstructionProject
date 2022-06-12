close all;
clear all;
clc;

%% Model- and Learning-Based Inverse Problems in Imaging FS22
% Group project
% Group members: Zhexin Wu, Cristina Almagro-Pérez, Runpu Hao
% Code author: Runpu Hao
% Data modified: 10/06/2022

% Locally low-rank reconstruction
% Comparison of different acceleration rates, outputs a comparison image
% for a randomly chosen patient for visualization
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

%% show data for a certain patient as illustration
img_p = imgs(:,:,:,1);

%% Iterate over different acceleration rates
acceleration_R = [5, 10, 15, 20, 25];
figure();
subplot(2, 3, 1)
imagesc(abs(img_p(:,:,1)));
title('ground truth', 'FontSize', 12);
for i_accR = 1:numel(acceleration_R)
    %% generate undersampling masks
    mask = rand(N, 1, Nt) < 1 / acceleration_R(i_accR);
    mask(floor(end/2)-5:floor(end/2)+5, :, :) = 1;
    % sample center of k-space densely (keep central 10 scanning lines)

    %% simulate acquisition, add reconstruction noise
    % set noise standard deviation to fixed 0.01
    noise_std = 0.01;
    kspc = i2k(img_p .* coils, 1);
    kspc = kspc + noise_std * ( randn(size(kspc)) + 1j * randn(size(kspc)));
    kspc = kspc .* mask;  
    %% zero-filled reconstruction
    rec_zf = sum(conj(coils) .* k2i(kspc, 1), 4) ./ sum(abs(coils).^2, 4);
    %% locally low rank reconstruction
    rec_pgd = pgd_nuclear(kspc, coils, mask,  0.3, 12, 100);
    %% plot the images
    subplot(2,3,i_accR+1)
    imagesc(abs(rec_pgd(:,:,1)));
    title(['LLR reconstruction, acc\_rate = ', num2str(acceleration_R(i_accR))], 'FontSize', 12);
end

