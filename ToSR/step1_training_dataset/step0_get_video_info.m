src_dir='I:\CDVL134\320_240';
VideoDir = dir([src_dir '\*.avi']);
fid = fopen('data_info.txt','w');
for j =1:length(VideoDir)
    fileName = [src_dir '\' VideoDir(j).name];
    Read_obj = VideoReader(fileName);
    numFrames = Read_obj.NumberOfFrames;
    hei=Read_obj.Height;
    wid=Read_obj.Width;
    fprintf(fid,'%s    %d    %d    %d\n', VideoDir(j).name,numFrames,hei,wid);
end
fclose(fid);