close all;
clear all;
clc;

%% Model- and Learning-Based Inverse Problems in Imaging FS22
% Group project
% Group members: Zhexin Wu, Cristina Almagro-Pérez, Runpu Hao
% Code author: Runpu Hao
% Data modified: 10/06/2022

% Locally low-rank reconstruction
% Comparison of different noise standard deviations, outputs a comparison
% image for a randomly chosen patient for visualization
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

%% Iterate over different noise standard deviations
noise_std = [0.01, 0.02, 0.03, 0.04, 0.05];
figure();
subplot(2, 3, 1)
imagesc(abs(img_p(:,:,1)));
title('ground truth', 'FontSize', 12);
for i_noise = 1:numel(noise_std)
    %% generate undersampling masks
    % set acceleration rate to fixed 5 times
    acceleration_R = 5;
    mask = rand(N, 1, Nt) < 1 / acceleration_R;
    mask(floor(end/2)-5:floor(end/2)+5, :, :) = 1;
    % sample center of k-space densely (keep central 10 scanning lines)
    %% simulate acquisition, add reconstruction noise
    % Vary noise standard deviation in each iteration
    noise_std_add = noise_std(i_noise);
    kspc = i2k(img_p .* coils, 1);
    kspc = kspc + noise_std_add * ( randn(size(kspc)) + 1j * randn(size(kspc)));
    kspc = kspc .* mask;  
    %% zero-filled reconstruction
    rec_zf = sum(conj(coils) .* k2i(kspc, 1), 4) ./ sum(abs(coils).^2, 4);
    %% locally low rank reconstruction
    rec_pgd = pgd_nuclear(kspc, coils, mask,  0.3, 12, 100);
    %% plot the images
    subplot(2,3,i_noise+1)
    imagesc(abs(rec_pgd(:,:,1)));
    title(['LLR reconstruction, noise\_std = ', num2str(noise_std(i_noise))], 'FontSize', 12);
end

