% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
fprintf('In directory %s, %d non-face images are found, ', non_face_scn_path, num_images);

% placeholder to be deleted
features_neg = zeros(0, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);

cell_num = feature_params.template_size / feature_params.hog_cell_size;
for img_i = 1 : num_images
    full_dir = fullfile( non_face_scn_path, image_files(img_i).name);
    img_color = imread(full_dir);
    img_gray = rgb2gray(img_color);
    img_single = im2single(img_gray);    % convert image to single precision, e.g. 255 to 1.0
    height = size(img_gray, 1);
    width = size(img_gray, 2);
    
    hog = vl_hog(img_single, feature_params.hog_cell_size);
    
    for left = 1 : cell_num : (size(hog, 2) - cell_num + 1)
        for top = 1 : cell_num : (size(hog, 1) - cell_num + 1)
            hog_in_window = hog(top : top + cell_num - 1, left : left + cell_num - 1, :);
            hog_in_window = reshape(hog_in_window, 1, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
            features_neg = [features_neg; hog_in_window];
        end
    end
end
fprintf('%d non-face templates are sampled.\n', length(features_neg));