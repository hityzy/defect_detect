close all;
clear;
clc

curt_dir = fileparts(fileparts(mfilename('fullpath')));
cd (curt_dir);

run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup')); %fileparts = ../
caffe_path = ans;
%%
class_idx = input('please input class_idx:','s');%'class2';%'class7';%change different classes of defect
class = ['class',class_idx];

opts = defect_detect_init(class , caffe_path);
%% calcalate average image
average_image = cal_avg_img(opts ,  curt_dir , class);

%% preparing training data
if not(exist(fullfile(curt_dir , 'output' , 'images_patch' , class , 'neg_data_train.mat') , 'file'))&&not(exist(fullfile(curt_dir , 'output' ,'images_patch' ,  class ,'pos_data_train.mat') , 'file'))
    [pos_data_train , neg_data_train ]  = defect_get_train_data( opts ,  curt_dir , class , average_image);
else
    [ neg_data_train , pos_data_train , average_image] = load_train_data(curt_dir , class);
end
%%

opts.feature_map_size=size(pos_data_train , 1)/opts.scale;

%% preparing validation data
if not(exist(fullfile(curt_dir , 'output' , 'images_patch' , class, 'neg_data_test.mat') , 'file'))&&not(exist(fullfile(curt_dir , 'output' ,'images_patch' , class ,'pos_data_test.mat') , 'file'))
    [pos_data_test , neg_data_test]  = defect_get_test_data( opts ,  curt_dir , average_image , class);
else
    [ neg_data_test , pos_data_test ] = load_test_data( curt_dir , class );
end
%%
defect_finetune_hnm(opts , pos_data_train , neg_data_train , pos_data_test , neg_data_test);

