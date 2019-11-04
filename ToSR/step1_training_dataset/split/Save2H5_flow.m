function Save2H5_flow(flow1,flow2,batch_size,filename,i)
count=size(flow1,4);
chunksz = batch_size;
created_flag = false;
totalct = 0;
for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = flow1(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = flow2(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5([filename(1:end-3),'_split',num2str(i),'.h5'], batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp([filename(1:end-3),'_split',num2str(i),'.h5'])