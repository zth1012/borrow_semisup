function show_montage(img_files, row_res, col_res)

if nargin == 1
    row_res = 200;
    col_res = 200;
end

images = cellfun(@(x) imread(x), img_files, 'UniformOutput',false);
images = cellfun(@(x) imresize(x, [row_res, col_res]), images, 'UniformOutput',false);

num_img = length(img_files);
I = zeros(row_res, col_res, 3, num_img);
for i = 1 : num_img
    I(:,:,:,i) = images{i};
end

% montage(I, 'Size', [1,length(img_files)]);
montage(uint8(I));