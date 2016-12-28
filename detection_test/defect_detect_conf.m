function conf  = defect_detect_conf( class , caffe_path )
%%
  conf.net1_path = 'models/defect_test1.prototxt';
  conf.weights_path = fullfile(pwd , 'output' , 'train_models' , class,  [class , '_final']);
  
  conf.batch_size = 256;
  
  if (strcmp(class,'class3')||strcmp(class,'class4'))
      conf.patch_size = 64;
      conf.ext = 'png';
  elseif (strcmp(class,'class6'))
      conf.patch_size = 64;
      conf.ext = 'bmp';
  elseif (strcmp(class,'class7'))
      conf.patch_size = 32;
      conf.ext = 'bmp';
  else%need to be defined
      conf.patch_size = 32;
      conf.ext = 'png';
  end
  
  conf.stride = conf.patch_size/2;
  
  conf.pos_thresh = 0.98;
%% caffe net init
  conf.useGpu = true;


  if (conf.useGpu==true)
       conf.gpu_id = 1;%auto_select_gpu;
       conf.caffe_version = 'faster_rcnn_external';
       active_caffe_mex(conf.gpu_id , caffe_path);
  else
       caffe.reset_all();
       caffe.set_mode_cpu();
  end 
  conf.net1 = caffe.Net(conf.net1_path, 'test');
  conf.net1.copy_from(conf.weights_path);
  
end

