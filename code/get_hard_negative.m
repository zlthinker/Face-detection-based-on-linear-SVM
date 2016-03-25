function hard_neg = get_hard_negative(non_face_scn_path, w, b, feature_params)

test_scenes = dir( fullfile( non_face_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
hard_neg = [];

% parameter
threshold = 1;
sample_acceleration = 0.7;
cell_num = feature_params.template_size / feature_params.hog_cell_size;


for j = 1:length(test_scenes)     
    img = imread( fullfile( non_face_scn_path, test_scenes(j).name ));
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    img = im2single(img);
    
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_neg = zeros(0, cell_num^2 * 31);
    
    sample_rate = 1.0;
    while min(size(img, 1), size(img, 2)) * sample_rate >= feature_params.template_size
        downsample = imresize(img, sample_rate);
        %convert hog to hog space
        hog = vl_hog(downsample, feature_params.hog_cell_size);
        
        for left = 1:(size(hog, 2) - cell_num + 1)
            for top = 1:(size(hog, 1) - cell_num + 1)
                hog_in_window = hog(top : top + cell_num - 1, left : left + cell_num - 1, :);
                hog_in_window = reshape(hog_in_window, 1, cell_num^2 * 31);
                confidence = hog_in_window * w + b;
                if (confidence > threshold)
                    left_pixel = (left - 1) * feature_params.hog_cell_size + 1;
                    top_pixel = (top - 1) * feature_params.hog_cell_size + 1;
                    right_pixel = left_pixel + feature_params.template_size - 1;
                    bottom_pixel = top_pixel + feature_params.template_size - 1;
                    box = [left_pixel / sample_rate, top_pixel / sample_rate, right_pixel / sample_rate, bottom_pixel / sample_rate];
                    cur_bboxes = [cur_bboxes; box];
                    cur_confidences = [cur_confidences; confidence];
                    cur_neg = [cur_neg; hog_in_window];
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
    cur_neg   = cur_neg(  is_maximum,:);

    hard_neg = [hard_neg; cur_neg];
end

fprintf('After hard negative mining, %d false positives are found.\n', size(hard_neg, 1));
