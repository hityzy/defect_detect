function [ opts ] = defect_detect_init(class , caffe_path)
%% File Path
     opts.caffe_solver_path = 'models/solver_ZF.prototxt';
     opts.caffe_net_path = 'models/defect_train_val.prototxt';
     opts.caffe_init_weights_path = 'models/proposal_final_ZF.caffemodel';
     mkdir_if_missing(fullfile('output' , 'train_models' , class));
     opts.model_path = fullfile('output' , 'train_models' , class);
     
     mkdir_if_missing(fullfile('output' , 'caffe_log' , class , 'caffe_log'));
     opts.caffe_init_path = fullfile('output' , 'caffe_log' , class , 'caffe_log');
     %% diary path
     timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
      
     mkdir_if_missing(fullfile('output' , 'diary' , class));
     opts.diary_path_loss = fullfile('output' , 'diary' , class , [timestamp , '_loss.txt']);
     opts.fid_loss = fopen( opts.diary_path_loss , 'w+');
     
     opts.diary_path_accurancy = fullfile('output' , 'diary' , class ,[timestamp , '_accurancy.txt']);
     opts.fid_accurancy = fopen( opts.diary_path_accurancy , 'w+');
      %% data path
     opts.data_path = fullfile( 'dataset' , [class , '_def']);
     opts.train_img_path = fullfile(opts.data_path , [class , '_def_img'] , 'train');
     opts.train_data_path = fullfile(opts.data_path , 'train');
     opts.test_data_path = fullfile(opts.data_path , 'test');
%% train parameters
    opts.batch_size = 128; % reduce it in case of out of gpu memory
    opts.batch_pos = opts.batch_size/4;
    opts.batch_neg = opts.batch_size/4*3;
    
    opts.batchSize_hnm = 128;%256
    opts.batchAcc_hnm = 4;
 
    opts.maxiter_all = 300;%30;
    
    opts.do_val = 1;
    opts.batch_size_val = 128;
    
    if strcmp(class , 'class6')||strcmp(class , 'class7')
         opts.ext = 'bmp';
    else
        opts.ext = 'png';
    end

%% caffe init
    opts.rng_seed=6;
    opts.useGpu = true;
    
    if (opts.useGpu==true)
        opts.gpu_id = 1;%auto_select_gpu;
        opts.caffe_version = 'caffe_master';
        active_caffe_mex(opts.gpu_id , caffe_path);
    else
        caffe.reset_all();
        caffe.set_mode_cpu();
    end    
    
    caffe.init_log(opts.caffe_init_path);
%% caffe solver init
    opts.caffe_solver = caffe.Solver(opts.caffe_solver_path);
    opts.caffe_solver.net.copy_from(opts.caffe_init_weights_path);
    
    input_size =   size(opts.caffe_solver.net.blobs('data').get_data() , 1);
    output_size = size(opts.caffe_solver.net.blobs('proposal_cls_prob').get_data() , 1);
    opts.scale = input_size/output_size;
    
%%


end

