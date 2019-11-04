%%%%%%%%%%%%%%%%%%%%%%%    split2patch,check warp then downsample       %%%%%%%%%%%%%
clear;
%set
% phase='valid';
phase='train';
patch_size=128;
sliding_stride_wid=40;
sliding_stride_hei=30;
select_num=10;
batch_size=16;

%% get namelist
% list_frame1=importdata( './step4_valid_frame1_list.txt');
% list_frame2=importdata( './step4_valid_frame2_list.txt');
% list_flow=importdata( './step4_valid_flow_list.txt');
list_frame1=importdata( ['./step4_',phase,'_frame1_list.txt']);
list_frame2=importdata( ['./step4_',phase,'_frame2_list.txt']);
list_flow=importdata( ['./step4_',phase,'_flow_list.txt']);
list=list_frame1;
%% split to patch
%{
data1=zeros(patch_size/4,patch_size/4,3,length(list)*select_num*7);
label1=zeros(patch_size,patch_size,3,length(list)*select_num);
data2=zeros(patch_size/4,patch_size/4,3,length(list)*select_num*7);
label2=zeros(patch_size,patch_size,3,length(list)*select_num);
flow1=zeros(patch_size,patch_size,1,length(list)*select_num);
flow2=zeros(patch_size,patch_size,1,length(list)*select_num);
%}
data1=zeros(patch_size/4,patch_size/4,3,1);
label1=zeros(patch_size,patch_size,3,1);
data2=zeros(patch_size/4,patch_size/4,3,1);
label2=zeros(patch_size,patch_size,3,1);
flow1=zeros(patch_size,patch_size,1,1);
flow2=zeros(patch_size,patch_size,1,1);
num=0;
exclude_num=0;
for idx = 1 : length(list)
    %flow for i-th frame
    flow_tvl1_x=readMatFlow(list_flow{2*idx-1});
    flow_tvl1_y=readMatFlow(list_flow{2*idx});
    flow=flow_tvl1_x;
    flow(:,:,2)=flow_tvl1_y;
    % RGB frame sequence
    hr_seq1=find_sequence(list_frame1{idx});
    hr_seq2=find_sequence(list_frame2{idx});
    
    % select the patch with large motion (save left above point)
    im_tvl1=sqrt(flow_tvl1_x.*flow_tvl1_x+flow_tvl1_y.*flow_tvl1_y);
    large_tvl1=select_large_motion(im_tvl1,patch_size,sliding_stride_wid,sliding_stride_hei,select_num);
   
    
    %% crop and downsample
    for k=1:select_num
        %crop
        point=large_tvl1{k};
        hr_patch_seq1=hr_seq1(point(1):point(1)+patch_size-1,point(2):point(2)+patch_size-1,:,:);
        hr_patch_seq2=hr_seq2(point(1):point(1)+patch_size-1,point(2):point(2)+patch_size-1,:,:);
        flow_patch=flow(point(1):point(1)+patch_size-1,point(2):point(2)+patch_size-1,:);
        
        if ~CheckWarp(hr_patch_seq1(:,:,:,4),hr_patch_seq2(:,:,:,4),flow_patch)
            exclude_num=exclude_num+1;
            continue;
        end
        num=num+1;
        %set downsample and 'save'
        lr_patch_seq1=imresize(hr_patch_seq1,1/4, 'bicubic');
        lr_patch_seq2=imresize(hr_patch_seq2,1/4, 'bicubic');
        
        data1(:,:,:,(num-1)*7+1:num*7)=lr_patch_seq1;
        label1(:,:,:,num)=hr_patch_seq1(:,:,:,4);
        data2(:,:,:,(num-1)*7+1:num*7)=lr_patch_seq2;
        label2(:,:,:,num)=hr_patch_seq2(:,:,:,4);
        % data1(:,:,:,(num-1)*7+1:num*7)=double(lr_patch_seq1)/255.0;
        % label1(:,:,:,num)=double(hr_patch_seq1(:,:,:,4))/255.0;
        % data2(:,:,:,(num-1)*7+1:num*7)=double(lr_patch_seq2)/255.0;
        % label2(:,:,:,num)=double(hr_patch_seq2(:,:,:,4))/255.0;
        flow1(:,:,1,num)=flow_patch(:,:,1);
        flow2(:,:,1,num)=flow_patch(:,:,2);
    end
end

count=size(data1,4);
chunksz1 = batch_size*7;
chunksz2 = batch_size;
created_flag = false;
% totalct = 0;
curr_dat_sz=0;
curr_lab_sz=0;
for batchno = 1:floor(count/chunksz1)
    last_read1=(batchno-1)*chunksz1;
    last_read2=(batchno-1)*chunksz2;
    batchdata = data1(:,:,:,last_read1+1:last_read1+chunksz1); 
    batchlabs = label1(:,:,:,last_read2+1:last_read2+chunksz2);

    startloc = struct('dat',[1,1,1,curr_dat_sz(end)+1], 'lab', [1,1,1,curr_lab_sz(end)+1]);
%     curr_dat_sz = store2hdf5_my('test.h5', batchdata, batchlabs, ~created_flag, startloc, chunksz2); 
    [curr_dat_sz, curr_lab_sz]= store2hdf5_my([phase,'_frame1.h5'], batchdata, batchlabs, ~created_flag, startloc, chunksz2);
    created_flag = true;
%     totalct = curr_dat_sz(end);
end

count=size(data2,4);
chunksz1 = batch_size*7;
chunksz2 = batch_size;
created_flag = false;
% totalct = 0;
curr_dat_sz=0;
curr_lab_sz=0;
for batchno = 1:floor(count/chunksz1)
    last_read1=(batchno-1)*chunksz1;
    last_read2=(batchno-1)*chunksz2;
    batchdata = data2(:,:,:,last_read1+1:last_read1+chunksz1); 
    batchlabs = label2(:,:,:,last_read2+1:last_read2+chunksz2);

    startloc = struct('dat',[1,1,1,curr_dat_sz(end)+1], 'lab', [1,1,1,curr_lab_sz(end)+1]);
%     curr_dat_sz = store2hdf5_my('test.h5', batchdata, batchlabs, ~created_flag, startloc, chunksz2); 
    [curr_dat_sz, curr_lab_sz]= store2hdf5_my([phase,'_frame2.h5'], batchdata, batchlabs, ~created_flag, startloc, chunksz2);
    created_flag = true;
%     totalct = curr_dat_sz(end);
end

count=size(flow1,4);
chunksz = batch_size;
created_flag = false;
totalct = 0;
for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = flow1(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = flow2(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5([phase,'_flow.h5'], batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp([phase,'_frame1.h5']);
h5disp([phase,'_frame2.h5']);
h5disp([phase,'_flow.h5']);