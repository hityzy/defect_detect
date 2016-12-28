function [ ims ] = im_crop_regions( img , rectangles ,class)
%%
crop_size = rectangles(1,3)+1;
num_boxes = size(rectangles , 1);
% imgs = repmat(img,[1,1,1,num_boxes]);
 ims = zeros(crop_size, crop_size, 3, num_boxes, 'single');
% crop = gpuArray(zeros(crop_size , crop_size , 3  , 'single'));
mat = load(fullfile(pwd , 'output' , 'images_patch' , class ,'average_image.mat'));
average_image = mat.average_image;
for i = 1:num_boxes
    rectangle = rectangles(i,:);
    if length(find(img(rectangle(2):(rectangle(2)+rectangle(4)) , rectangle(1):(rectangle(1)+rectangle(3)) , : )>=250))>=(crop_size^2*3)
        continue;
    end
%     crop = imcrop(img , rectangle);
    ims(:,:,:,i)  = img(rectangle(2):(rectangle(2)+rectangle(4)) , rectangle(1):(rectangle(1)+rectangle(3)) , : );
end

ims = bsxfun(@minus , ims , average_image);
end
%%

% crop_size = rectangles(1,3)+1;
% num_boxes = size(rectangles , 1);
% img = gpuArray(img);
% ims = gpuArray(zeros(crop_size, crop_size, 3, num_boxes, 'single'));
% 
% % ims = (zeros(crop_size, crop_size, 3, num_boxes, 'single'));
% 
% for i = 1:num_boxes
%     rectangle = rectangles(i,:);
%     [Xq,Yq,Zq] = meshgrid(linspace(rectangle(1),rectangle(1)+rectangle(3),crop_size), linspace(rectangle(2),rectangle(2)+rectangle(4),crop_size), 1:3);
%     ims(:,:,:,i) = interp3(img,Xq,Yq,Zq);
% end
%     ims = gather(ims);
% end
