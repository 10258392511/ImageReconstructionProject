function img = blocks2im_2D(blocks, bsz, sz, szn)
% blocks: bsz^2 - Nt - NUM_OF_BLOCKS

if numel(bsz) == 1
    bsz = [1,1] * bsz;
end

blocks = reshape(blocks, [bsz(1), bsz(2), szn(3), szn(1)/bsz(1), szn(2)/bsz(2)]);
blocks = permute(blocks, [1, 4, 2, 5, 3]);
img = reshape(blocks, szn);
img = img(1:sz(1), 1:sz(2), :);
end