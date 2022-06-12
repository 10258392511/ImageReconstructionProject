function [blocks, sz, szn] = im2blocks_2D(img, bsz)
% img: sz0-sz1-Nt
% bsz: scalar block size
% out
% blocks: bsz^2 - Nt - NUM_OF_BLOCKS
if numel(bsz) == 1
    bsz = [1,1] * bsz;
end

sz = size(img);
sz = sz(1:2);
pads = bsz.*ceil(sz ./ bsz) - sz;
% img = padarray(img, [pads, 0],  'symmetric', 'post');
if any(pads)
    img = padarray(img, [pads, 0],  'symmetric', 'post');
end

szn = size(img);
if numel(szn) == 2
    szn = [szn, 1];
end
                %    1       2        3      4         5      
img = reshape(img, [bsz(1), szn(1)/bsz(1), bsz(2), szn(2)/bsz(2),  szn(3)]);
blocks = permute(img, [1, 3, 5, 2, 4]);
blocks = reshape(blocks, prod(bsz), szn(3), []);
end