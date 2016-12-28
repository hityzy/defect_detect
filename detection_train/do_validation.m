function [ accurancy_fg , accurancy_bg ] = do_validation( opts , pos_data_test , neg_data_test )

%% do validation fg
    n_fg = size(pos_data_test,4);
    nBatches = ceil(n_fg/opts.batch_size_val);
    result=0;
    for i=1:nBatches

        batch = pos_data_test(:,:,:,opts.batch_size_val*(i-1)+1:min(end,opts.batch_size_val*i));
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
        batch = permute(batch, [2, 1, 3, 4]);
        batch = single(batch);


        label_tmp = rand( opts.feature_map_size , opts.feature_map_size , 1 , size(batch , 4));
        weight_tmp = rand(size(label_tmp));
        net_inputs = {batch, label_tmp, weight_tmp};

    % Reshape net's input blobs
        opts.caffe_solver.net.reshape_as_input(net_inputs);
        opts.caffe_solver.net.forward(net_inputs);
        prob = opts.caffe_solver.net.blobs('proposal_cls_prob').get_data();
%         res = squeeze(sum(sum(res ,1) , 2)) / (opts.feature_map_size(1)*opts.feature_map_size(2));
        score_fg = prob(:,:,1,:);
        score_fg = squeeze(sum(sum(score_fg ,1) , 2)) / (opts.feature_map_size*opts.feature_map_size);
        
        score_bg = prob(:,:,2,:);
        score_bg = squeeze(sum(sum(score_bg ,1) , 2)) / (opts.feature_map_size*opts.feature_map_size);
    
        res_fg = score_fg>score_bg;
        result = result+sum(res_fg);
    end
         accurancy_fg = result/n_fg;
%%

%% do validation bg
    n_bg = size(neg_data_test,4);
    nBatches = ceil(n_bg/opts.batch_size_val);
    result=0;
    for i=1:nBatches

        batch = neg_data_test(:,:,:,opts.batch_size_val*(i-1)+1:min(end,opts.batch_size_val*i));
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
        batch = permute(batch, [2, 1, 3, 4]);
        batch = single(batch);


        label_tmp = rand( opts.feature_map_size , opts.feature_map_size , 1 , size(batch , 4));
        weight_tmp = rand(size(label_tmp));
        net_inputs = {batch, label_tmp, weight_tmp};

    % Reshape net's input blobs
        opts.caffe_solver.net.reshape_as_input(net_inputs);
        opts.caffe_solver.net.forward(net_inputs);
        prob = opts.caffe_solver.net.blobs('proposal_cls_prob').get_data();
%         res = squeeze(sum(sum(res ,1) , 2)) / (opts.feature_map_size(1)*opts.feature_map_size(2));
        score_fg = prob(:,:,1,:);
        score_fg = squeeze(sum(sum(score_fg ,1) , 2)) / (opts.feature_map_size*opts.feature_map_size);
        
        score_bg = prob(:,:,2,:);
        score_bg = squeeze(sum(sum(score_bg ,1) , 2)) / (opts.feature_map_size*opts.feature_map_size);
    
        res_bg = score_fg<score_bg;
        result = result+sum(res_bg);
        
    end
    accurancy_bg = result/n_bg;
        
%%


end

