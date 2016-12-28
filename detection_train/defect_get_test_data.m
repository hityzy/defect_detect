function  [total_pos_patchs , total_neg_patchs]  = defect_get_test_data( opts , curt_dir , average_image , class)

total_pos_patchs = [];
total_neg_patchs = [];

total_test_data = dir(opts.test_data_path);
total_test_data(1,:)=[];
total_test_data(1,:)=[];
num_total_test_data = size(total_test_data,1);

%%
for dir_idx = 1:num_total_test_data
%     pos_data_per_img = dir(fullfile(opts.test_data_path , total_test_data(dir_idx).name , 'pos' , '*png'));
%     neg_data_per_img = dir(fullfile(opts.test_data_path , total_test_data(dir_idx).name , 'neg' , '*png'));
    pos_data_per_img = dir(fullfile(opts.test_data_path , total_test_data(dir_idx).name , 'pos' , ['*.' , opts.ext]));
    neg_data_per_img = dir(fullfile(opts.test_data_path , total_test_data(dir_idx).name , 'neg' , ['*.' , opts.ext]));
    fprintf(['praparing validation data in ' , total_test_data(dir_idx).name , '\n']);
   %% prepare pos data per dir
    for img_idx = 1:size(pos_data_per_img,1)
        
        img_patch = single(imread(fullfile(opts.test_data_path , total_test_data(dir_idx).name , 'pos' , pos_data_per_img(img_idx).name)));
        if size(img_patch,3)==1
            img_patch = cat(3 , img_patch , img_patch , img_patch);
        end
        
        total_pos_patchs = cat(4,total_pos_patchs,img_patch);
    end
    %% prepare neg data per dir
    for img_idx = 1:size(neg_data_per_img,1)
        
        img_patch = single(imread(fullfile(opts.test_data_path , total_test_data(dir_idx).name , 'neg' , neg_data_per_img(img_idx).name)));
        if size(img_patch,3)==1
            img_patch = cat(3 , img_patch , img_patch , img_patch);
        end
        
        total_neg_patchs = cat(4,total_neg_patchs,img_patch);
    end
    %%
end

total_pos_patchs =bsxfun(@minus ,total_pos_patchs , average_image);
total_neg_patchs =bsxfun(@minus ,total_neg_patchs , average_image);

mkdir_if_missing(fullfile(curt_dir , 'output' , 'images_patch' ,class));
save(fullfile(curt_dir , 'output' , 'images_patch' , class,'neg_data_test.mat') , 'total_neg_patchs');
save(fullfile(curt_dir , 'output' ,'images_patch' , class,'pos_data_test.mat') , 'total_pos_patchs');
%%
end

