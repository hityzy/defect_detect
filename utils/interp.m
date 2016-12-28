im = im-128;
im = gpuArray(single(im));
ims = gpuArray(zeros(crop_size, crop_size, 3, num_boxes, 'single'));
window = gpuArray(zeros(crop_size, crop_size, 3, 'single'));

for i = 1:num_boxes;
    pad_h  = pad_y1(i);
    pad_w = pad_x1(i);
    
    if pad_h + crop_height(i) > crop_size
        crop_height(i) = crop_size - pad_h;
    end
    if pad_w + crop_width(i) > crop_size
        crop_width(i) = crop_size - pad_w;
    end% padding > 0 || square
    
     bboxes = boxes(i,:);
     [Xq,Yq,Zq] = meshgrid(linspace(bboxes(1),bboxes(3),crop_width(i)), linspace(bboxes(2),bboxes(4),crop_height(i)), 1:3);
     window(pad_h+(1:crop_height(i)), pad_w+(1:crop_width(i)), :) = interp3(im,Xq,Yq,Zq);
     ims(:,:,:,i) =window;%interp3(im,Xq,Yq,Zq);
end
ims = gather(ims);
end
