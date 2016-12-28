function [ neg_data , pos_data ] = load_test_data( curt_dir  , class)
    mat = load(fullfile(curt_dir , 'output' , 'images_patch' , class , 'neg_data_test.mat'));
    neg_data = mat.total_neg_patchs;
    mat = load(fullfile(curt_dir , 'output' , 'images_patch' , class ,'pos_data_test.mat'));
    pos_data = mat.total_pos_patchs;


%     average_image = sum(neg_data , 4);
%     average_image = average_image+sum(pos_data , 4);
%     average_image = average_image./(size(pos_data , 4)+size(neg_data , 4));
    
%     pos_data =bsxfun(@minus ,pos_data , average_image);
%     neg_data =bsxfun(@minus ,neg_data , average_image);
    
end

