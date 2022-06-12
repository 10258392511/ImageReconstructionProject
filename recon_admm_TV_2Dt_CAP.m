function [rec, itinfo] = recon_admm_TV_2Dt_CAP(d, mask, lambda, w, rho, maxiter, cg_iters, fft_dims)
% min F(X) = 1/2|M * F * Coils * X  - b|_2^2 + lambda * |D X|_1
% d - the k-space  sz1-sz2-Nt-Nimgs 
% mask should be broadcastable to size of d
% lambda - regularization coefficient
% w - TV weights for spatial and temporal components
% rho - augmentation parameter
% maxiter - maximal number of iteration
% cg_iters - number of iterations for x subproblem
%
% returns 
%   rec: reconstructed image of size sz1-sz2-Nt
%   itinfo: structure with optimization trace 
%       itinfo.fvals -- cost function value evaluated for x
%       itinfo.f_reg -- regularization penalty value
%       itinfo.f_data -- data residual
%       itinof.primal_r -- primal residual |Dx-z|

tau = 1.05;

itinfo = [];
itinfo.fvals = [];
itinfo.f_reg = [];
itinfo.f_data = [];
itinfo.primal_r = [];
 
%% Variable initialization

Fb = k2i(mask.*d,fft_dims);
x=Fb;

Dx = grad_op(x,w);
z = Dx;
u = z - Dx;
szx = size(x);

%% Updating parameters
for iter = 1:maxiter
    rho = rho * tau;

    %% X - update
    % solves min_x .5 * |M*F*C*x - b|^2 + rho/2 |z - D*x + u|^2
    mat_op = @(x) sq_op(x, mask, rho, szx, w,fft_dims); %this fft_dims is new
    rhs = Fb + rho * adj_grad_op(z + u,w); 

    [x,~,~,~,~] = pcg(mat_op, rhs(:), 1e-8, cg_iters, [], [], x(:));
    x = reshape(x, szx);
    
    Dx = grad_op(x,w);

    %% Z - update
    % solves min_Z lambda*|z|_1 + rho/2 |z - D*x + u|^2
    %val_f_z(iter)=sum(abs(z),'all')+ rho/2*(sum((abs(z - Dx + u)).^2, 'all'));
    tmp = Dx - u;
    z = proximal_l1(tmp, lambda/rho);
    %val_f_z_new(iter)=sum(abs(z),'all')+ rho/2*(sum((abs(z - Dx + u)).^2, 'all'));
    %% Dual update
    u = u + z - Dx;
    
    %% Log progress
    res1 =  mask .* (i2k(x,fft_dims)  - d);
    f_data =  sum( abs(res1(:)).^2)/2;
    reg_cost = sum(abs(Dx(:)));
    itinfo.fvals(end+1) = (f_data + reg_cost * lambda);
    itinfo.f_reg(end+1) = reg_cost;
    itinfo.f_data(end+1) = f_data;
    
    Dx = grad_op(x,w);
    itinfo.primal_r(end+1) = norm(Dx(:) - z(:));
    
end
rec = x;
end

%% Additional functions
function y = sq_op(x, mask, rho, sz, w,fft_dims)
    % This function implements the following linear operator
    %  C' * F ' * M * F * C + rho * D' * D
    % for image x
    
    x = reshape(x, sz); % vector to image array
    
    % Compute C' * F ' * M * F * C * x
    a1 = k2i(mask .* i2k(x, fft_dims), fft_dims);
    % Compute rho * D' * D * x
    a2 = rho * adj_grad_op(grad_op(x,w),w);
    
    y = a1 + a2;
    y = y(:);
end


