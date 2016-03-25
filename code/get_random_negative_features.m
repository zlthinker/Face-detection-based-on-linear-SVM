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
fprintf('In directory %s, %d non-face images are found...\n', non_face_scn_path, num_images);

% placeholder to be deleted
features_neg = zeros(0, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);

cell_num = feature_params.template_size / feature_params.hog_cell_size;

%parameters
sample_acceleration = 0.7;

for img_i = 1 : num_images
    full_dir = fullfile( non_face_scn_path, image_files(img_i).name);
    fprintf('[%d]Sample from non-face image %s...\n', img_i, full_dir);
    img_color = imread(full_dir);
    img_gray = rgb2gray(img_color);
    img_single = im2single(img_gray);    % convert image to single precision, e.g. 255 to 1.0
    height = size(img_gray, 1);
    width = size(img_gray, 2);
    sample_rate = 1.0;
    
    while min(width, height) * sample_rate >= feature_params.template_size
        downsample = imresize(img_single, sample_rate);
        hog = vl_hog(downsample, feature_params.hog_cell_size);
        temp_h = fix(size(hog, 1) / cell_num);
        temp_w = fix(size(hog, 2) / cell_num);
        hog = hog(1:temp_h * cell_num, 1:temp_w*cell_num, :);
        height_divide = ones(1, temp_h) * cell_num;
        width_divide = ones(1, temp_w) * cell_num;
        templates = mat2cell(hog, height_divide, width_divide, 31);
        templates = reshape(templates, 1, temp_h * temp_w);
        step = 4;
%         disp(size(templates));
        for temp_i = 1:step:size(templates, 2)
            hog_in_window = reshape(templates{1, temp_i}, 1, cell_num^2 * 31);
            features_neg = [features_neg; hog_in_window];
        end
        sample_rate = sample_rate * sample_acceleration;
%         for left = 1 : cell_num * 3 : (size(hog, 2) - cell_num + 1)
%             for top = 1 : cell_num * 3 : (size(hog, 1) - cell_num + 1)
%                 hog_in_window = hog(top : top + cell_num - 1, left : left + cell_num - 1, :);
%                 hog_in_window = reshape(hog_in_window, 1, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
%                 features_neg = [features_neg; hog_in_window];
%             end
%         end
    end
end
fprintf('%d non-face templates are sampled.\n', size(features_neg, 1));