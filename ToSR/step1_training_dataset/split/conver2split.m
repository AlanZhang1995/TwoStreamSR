split_num=4;

filename='train_frame1.h5';
h5disp(filename)
data=h5read(filename,'/data');
label=h5read(filename,'/label');
total=size(label,4);
stride=total/split_num;
for i=1:split_num
    data_tmp=data(:,:,:,(i-1)*stride*7+1:i*stride*7);
    label_tmp=label(:,:,:,(i-1)*stride+1:i*stride);
    
    %save to H5
    Save2H5(data_tmp,label_tmp,16,filename,i);
end

%-------------------

filename='train_frame2.h5';
h5disp(filename)
data=h5read(filename,'/data');
label=h5read(filename,'/label');
total=size(label,4);
stride=total/split_num;
for i=1:split_num
    data_tmp=data(:,:,:,(i-1)*stride*7+1:i*stride*7);
    label_tmp=label(:,:,:,(i-1)*stride+1:i*stride);
    
    %save to H5
    Save2H5(data_tmp,label_tmp,16,filename,i);
end

%-------------------

filename='train_flow.h5';
h5disp(filename)
data=h5read(filename,'/data');
label=h5read(filename,'/label');
total=size(label,4);
stride=total/split_num;
for i=1:split_num
    data_tmp=data(:,:,:,(i-1)*stride+1:i*stride);
    label_tmp=label(:,:,:,(i-1)*stride+1:i*stride);
    
    %save to H5
    Save2H5_flow(data_tmp,label_tmp,16,filename,i);
end