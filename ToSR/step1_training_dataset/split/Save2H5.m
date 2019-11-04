function Save2H5(data1,label1,batch_size,filename,i)
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
    [curr_dat_sz, curr_lab_sz]= store2hdf5_my([filename(1:end-3),'_split',num2str(i),'.h5'], batchdata, batchlabs, ~created_flag, startloc, chunksz2);
    created_flag = true;
%     totalct = curr_dat_sz(end);
end
h5disp([filename(1:end-3),'_split',num2str(i),'.h5'])