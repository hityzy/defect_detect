close all;
clear;
clc

class_idx = input('please input class_idx:','s');%'class2';
class = ['class',class_idx];
curt_dir = fileparts(fileparts(mfilename('fullpath')));
cd (curt_dir);

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup')); %fileparts = ../
caffe_path = ans;

conf = defect_detect_conf(class , caffe_path);
images = dir(fullfile(pwd , 'dataset' , [class , '_def'] , [class , '_def_img'] , 'test' , ['*.' , conf.ext]));


for i = 1:length(images)
img = single(imread(fullfile(pwd , 'dataset' , [class , '_def'] , [class , '_def_img'] , 'test' , images(i).name)));
idx = find(images(i).name =='-');
imgname = images(i).name(idx+1:end);

if size(img,3)==1
    img = cat(3 , img , img , img);
end

if strcmp(class , 'class7')%guilded filter improve the performance of class7
   img = imguidedfilter(img,  'NeighborhoodSize' , [17,17] , 'DegreeOfSmoothing' , 110);
end
 

x = (1:conf.stride:size(img , 2)-(conf.patch_size-1))'*ones();
y = (1:conf.stride:size(img , 1)-(conf.patch_size-1))';
size_x = length(x);
size_y = length(y);
x = repmat(x , [1 , size_y])';
x = x(:);
y = repmat(y , [size_x , 1]);
w =(conf.patch_size-1)*ones(size(x)); 
h = (conf.patch_size-1)*ones(size(y));
rectangles =[x , y , w , h];

if strcmp(class , 'class6')
    idx0 = (rectangles(:,2)<=64) +(rectangles(:,2)>=(size(img , 1)-(conf.patch_size)-10));
    rectangles(idx0>0 , :)=[];
end

 tic;
ims = im_crop_regions(img , rectangles , class);
res_ = [];
nbatchs = ceil(size(ims , 4)/ conf.batch_size); 
for i = 1:nbatchs
       batch = ims(:,:,:,conf.batch_size*(i-1)+1:min(end,conf.batch_size*i));
       net_inputs = {batch};
% Reshape net's input blobs
       conf.net1.reshape_as_input(net_inputs);
       res = conf.net1.forward(net_inputs);
       res = res{1};
       [res_w , res_h,~ ,~] = size(res);
       res = sum(sum(res,1),2);
       res = squeeze(res(:,:,1 , :))/(res_w*res_h);
       res_ = [res_ ; res];
end
    defect_idx = find(res_>=conf.pos_thresh);
if isempty(defect_idx)
    defect_idx = find(res_>=conf.pos_thresh*0.9);
end
defect = rectangles(defect_idx , :);
score = res_(defect_idx);
time = toc;
fprintf([imgname ,' cost %fs...\n'] , time);


%% display results
score_map = zeros(size(img , 1) , size(img , 2) , 'single');
for i =1: size(defect , 1)
score_map(defect(i,2):(defect(i,2)+defect(i,4)) , defect(i,1):(defect(i,1)+defect(i,3))) = ...
    score_map(defect(i,2):(defect(i,2)+defect(i,4)) , defect(i,1):(defect(i,1)+defect(i,3)))+score(i);
end
 Omin = min(score_map(score_map>0));
 Omax = max(score_map(score_map>0));
 Nmax = 256;
 Nmin = 64;
 if not(Omin==Omax)
    score_map(score_map>0) = (Nmax-Nmin) / (Omax-Omin) *(score_map(score_map>0)-Omin)+Nmin;
 else
    score_map(score_map>0) =Nmax;
 end

imshow(uint8(img));
img_R = img(:,:,1);
img_R(score_map>0) = img_R(score_map>0)*1+score_map(score_map>0)*0.7;
img(:,:,1) = img_R;
waitforbuttonpress;
imshow(uint8(img));
text(10,10,imgname,'Color','r', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 10); 
waitforbuttonpress;


% %% another way to show results
% % imshow(uint8(img));
% % for i =1: size(defect , 1)
% % rectangle('Position', defect(i,:),  'EdgeColor', [0 0.7 0], 'Linewidth', 0.5);
% % end
% % waitforbuttonpress;
% %%
% 
% 
end

%% test*********************************************************************************
% i = 121;
% img = single(imread(fullfile(pwd , 'dataset' , 'class1_def' , 'class1_def_img' , 'test' , sprintf('class1_def-%d.png',i))))-69.3;
% if size(img,3)==1
%     img = cat(3 , img , img , img);
% end
% x=1;
% y=1;
% w=511;
% h=511;
% imshow(uint8(img+69.3));
% rectangle('Position', [x,y,w,h], 'EdgeColor', [0 1 0], 'Linewidth', 2);
%  
% patch1 = imcrop(img , [x,y,w,h]);
% net_inputs = {patch1};
% % Reshape net's input blobs
% conf.net1.reshape_as_input(net_inputs);
% tic
% res = conf.net1.forward(net_inputs);
% toc
% 
% res = res{1};
% res = res(:,:,1);
% % score = sum(sum(res))/16
%%
