function [nRMSE] = nRMSE_comp(recon_img, gt_img)

size_img = size(recon_img);

diff = (recon_img - gt_img).^2;
sum_diff = sum(diff, 'all');
RMSE = sqrt(sum_diff/(size_img(1) * size_img(2) * size_img(3)));
nRMSE = RMSE/(max(recon_img, [], 'all')-min(recon_img, [], 'all'));
nRMSE = abs(nRMSE);

end

