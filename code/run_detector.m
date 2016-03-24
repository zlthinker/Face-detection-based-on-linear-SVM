% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
fprintf('To detect faces from %d images in %s...\n.', length(test_scenes), test_scn_path);

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

% parameter
threshold = 0.7;
sample_acceleration = 0.7;


for j = 1:length(test_scenes)
      
    fprintf('Detecting faces in %s\n', test_scenes(j).name)
    img = imread( fullfile( test_scn_path, test_scenes(j).name ));
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    img = im2single(img);
    
    %You can delete all of this below.
    % Let's create 15 random detections per image
    %cur_x_min = rand(15,1) * size(img,2);
    %cur_y_min = rand(15,1) * size(img,1);
    %cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
    %cur_confidences = rand(15,1) * 4 - 2; %confidences in the range [-2 2]
    %cur_image_ids(1:15,1) = {test_scenes(i).name};
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0, 1);

    cell_num = feature_params.template_size / feature_params.hog_cell_size;
    sample_rate = 1.0;
    while min(size(img, 1), size(img, 2)) * sample_rate >= feature_params.template_size
        downsample = imresize(img, sample_rate);
        %convert hog to hog space
        hog = vl_hog(downsample, feature_params.hog_cell_size);
        
        for left = 1:(size(hog, 2) - cell_num + 1)
            for top = 1:(size(hog, 1) - cell_num + 1)
                hog_in_window = hog(top : top + cell_num - 1, left : left + cell_num - 1, :);
                hog_in_window = reshape(hog_in_window, 1, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
                confidence = hog_in_window * w + b;
                if (confidence > threshold)
                    left_pixel = (left - 1) * feature_params.hog_cell_size + 1;
                    top_pixel = (top - 1) * feature_params.hog_cell_size + 1;
                    right_pixel = left_pixel + feature_params.template_size - 1;
                    bottom_pixel = top_pixel + feature_params.template_size - 1;
                    box = [left_pixel / sample_rate, top_pixel / sample_rate, right_pixel / sample_rate, bottom_pixel / sample_rate];
                    cur_bboxes = [cur_bboxes; box];
                    cur_confidences = [cur_confidences; confidence];
                    cur_image_ids = [cur_image_ids; test_scenes(j).name];
                end          
            end         
        end
        sample_rate = sample_rate * sample_acceleration;
               
    end
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.

    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);

    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];

end




