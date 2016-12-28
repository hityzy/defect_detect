function [ average_image ] = cal_avg_img( opts ,  curt_dir , class )

train_images = dir(fullfile(opts.train_img_path , ['*.' , opts.ext]));

num_train_images = size(train_images,1);

mkdir_if_missing(fullfile(curt_dir , 'output' , 'images_patch' ,class));
%%
if not(exist(fullfile(curt_dir , 'output' , 'images_patch' , class ,'average_image.mat') , 'file'))
%% if not exist average image
   for img_idx = 1:num_train_images
          img = single(imread(fullfile(opts.train_img_path , train_images(img_idx).name)));
          if  img_idx==1
              average_image = img;
              size_std = size(average_image);
          else
              if not(size(img , 1)==size_std(1)||size(img , 1)==size_std(1))
                  img = imresize(img , size_std);
              end
              average_image = average_image+img;
          end
   end
   
    average_image = average_image./num_train_images;
    average_image = mean(mean(average_image));
    save(fullfile(curt_dir , 'output' , 'images_patch' , class , 'average_image.mat') , 'average_image');

else
%% if exist average image
    mat = load(fullfile(curt_dir , 'output' , 'images_patch' , class ,'average_image.mat'));
    average_image = mat.average_image;
end

end

