function  [total_pos_patchs , total_neg_patchs , average_image]  = defect_get_train_data( opts , curt_dir ,class , average_image)

total_pos_patchs = [];
total_neg_patchs = [];

total_train_data = dir(opts.train_data_path);
total_train_data(1,:)=[];
total_train_data(1,:)=[];
num_total_train_data = size(total_train_data,1);

%%
for dir_idx = 1:num_total_train_data
    pos_data_per_img = dir(fullfile(opts.train_data_path , total_train_data(dir_idx).name , 'pos' , ['*.' , opts.ext]));
    neg_data_per_img = dir(fullfile(opts.train_data_path , total_train_data(dir_idx).name , 'neg' , ['*.' , opts.ext]));
    fprintf(['praparing train data in ' , total_train_data(dir_idx).name , '\n']);
   %% prepare pos data per dir
    for img_idx = 1:size(pos_data_per_img,1)
        
        img_patch = single(imread(fullfile(opts.train_data_path , total_train_data(dir_idx).name , 'pos' , pos_data_per_img(img_idx).name)));
        if size(img_patch,3)==1
            img_patch = cat(3 , img_patch , img_patch , img_patch);
        end
        
        total_pos_patchs = cat(4,total_pos_patchs,img_patch);
    end
    %% prepare neg data per dir
    for img_idx = 1:size(neg_data_per_img,1)
        
        img_patch = single(imread(fullfile(opts.train_data_path , total_train_data(dir_idx).name , 'neg' , neg_data_per_img(img_idx).name)));
        if size(img_patch,3)==1
            img_patch = cat(3 , img_patch , img_patch , img_patch);
        end
        
        total_neg_patchs = cat(4,total_neg_patchs,img_patch);
    end
    %%
end
%% data minus mean image
total_pos_patchs =bsxfun(@minus ,total_pos_patchs , average_image);
total_neg_patchs =bsxfun(@minus ,total_neg_patchs , average_image);
save(fullfile(curt_dir , 'output' , 'images_patch' , class , 'neg_data_train.mat') , 'total_neg_patchs');
save(fullfile(curt_dir , 'output' ,'images_patch' ,  class ,'pos_data_train.mat') , 'total_pos_patchs');

%%
end

