function [ neg_data , pos_data , average_image] = load_train_data( curt_dir , class )
    mat = load(fullfile(curt_dir , 'output' , 'images_patch' , class ,'neg_data_train.mat'));
    neg_data = mat.total_neg_patchs;
    mat = load(fullfile(curt_dir , 'output' , 'images_patch' , class ,'pos_data_train.mat'));
    pos_data = mat.total_pos_patchs;
    
    mat = load(fullfile(curt_dir , 'output' , 'images_patch' , class ,'average_image.mat'));
    average_image = mat.average_image;
% if not(exist(fullfile(curt_dir , 'output' , 'images_patch_32' , 'average_image.mat') , 'file'))
%     average_image = sum(neg_data , 4);
%     average_image = average_image+sum(pos_data , 4);
%     average_image = average_image./(size(pos_data , 4)+size(neg_data , 4));
%     save(fullfile(curt_dir , 'output' , 'images_patch_32' , 'average_image.mat') , 'average_image');
% else
%     mat = load(fullfile(curt_dir , 'output' , 'images_patch_32' , 'average_image.mat'));
%     average_image = mat.average_image;
% end      
%     pos_data =bsxfun(@minus ,pos_data , average_image);
%     neg_data =bsxfun(@minus ,neg_data , average_image);
    
end

