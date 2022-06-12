 function [x, itinfo] = pgd_nuclear(s, coils, mask, lam, bsz, maxiter)
% s: input k-space
% coils: coil sensitivities
% mask: undersampling mask
% lam: regularization strength
% bsz: patch size
% maxiter: number of iterations

% min F(X) = 1/2|M * F * Coils * X  - s|_2^2 + lam * \sum_i |P_i X|_*
% b - the k-space sz1-sz2-Nt-Ncoils for 2D
% coils size: sz1-sz2-1-Ncoils for 2D
% mask should be broadcastable to b
% lam - regularization coefficient
% bsz - block size, scalar integer

% initialize variables for gradient descent
tau = 0.95;
gd_steplen = 1; % GD step length
debug = false;

% initial estimate of the reconstruction
Fb = sum(conj(coils).*k2i(mask.*s, 1), 4);  
x = Fb ./ sum(abs(coils).^2, 4); 
% initialize with zero-filled solution (fill in missing frequencies with 0
% and apply an IFT)

% book keeping
itinfo = [];
itinfo.fvals = zeros(1, 1);
itinfo.f_nuc = zeros(1, 1);
itinfo.f_data = zeros(1, 1);


% main iterations of proximal gradient descent
for iter = 1 : maxiter
    % gradient of data term
    grad = sum(conj(coils).*k2i(mask.*i2k(coils.*x,1) - s, 1), 4); 
    
    % proximal map
    
    % gradient step x - alpha * grad
    xx = x - gd_steplen * grad;
    % proximal mapping on patches
    [x, nuc_cost] = prox_nuclear_patch(xx, bsz, gd_steplen * lam);
    
    % gradually decrease the step length
    gd_steplen = gd_steplen * tau;
    
    fdata = sum( abs(mask.*i2k(coils.*x,1) - s).^2, 'all')/2;
    itinfo.f_data(iter) = fdata;
    itinfo.fvals(iter) = fdata + lam * nuc_cost;
    itinfo.f_nuc(iter) = nuc_cost;
    
    if debug 
        subplot(1,2,1);
        imagesc(squeeze(abs(x(:, :, round(end/2)))));
        
        subplot(1,2,2);
        plot(log(itinfo.fvals), 'r.-');
        ylabel('logCost');
        pause(0.01);
    end 
end

end

function [x, nuc_cost] = prox_nuclear_patch(x, bsz, lam)
    % x: receives an image as an input as a 3D array
    % lam: parameter lam = alpha * lambda (see definition of proximal map
    % for nuclear norm)
    
    shifts = generate_random_shift(bsz * [1,1]); % generate random shift
    x = apply_shifts(x, shifts); % apply the random shift (circular shift of the array)
    
    % patchify image
    [C, sz1, sz2] = im2blocks_2D(x, bsz * [1,1]);
    
    % keep track of the cost
    nuc_cost = 0;
    
    % C: bsz^2 - Nt - Np: spatial dimension - temporal dimension - patch
    
    % dimension (how much patches do we have)
    for i = 1 : size(C, 3)   % iterate over all patches
        % calculate SVD
        [U, S, V] = svd(C(:, :, i), 'econ');
        
        % apply nuclear (l1) proximal map for matrices C(:,:, i) on
        % singular values
        prox_l1_map = proximal_l1(diag(S), lam);
        prox_final = U * diag(prox_l1_map) * V';
        
        % keep track if cost function is decreasing
        nuc_cost = nuc_cost + sum(prox_final(:));
        % construct back the porjected matrix
        C(:,:, i) = prox_final;
    end
    % assemble patches back to image
    x = blocks2im_2D(C, bsz, sz1, sz2);
    % shift back
    x = apply_shifts(x, -shifts);
    % compute the 
end

