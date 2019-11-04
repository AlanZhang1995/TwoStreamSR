%%%%%total is 31880 frames and there is 31880/40=797, so I config 40
%%%%%frames per video and there are 797 videos totally
list=importdata( './step1_train_image_list.txt');
FramePerVideo=40;
fps = 20;
for VideoNum=1:length(list)/FramePerVideo
    VideoName=['v_ApplyEyeMakeup_g01_c',num2str(VideoNum,'%.3d'),'.avi'];
    VideoPath=['I:\TmpVideoForDataset\',VideoName];
    
    aviobj=VideoWriter(VideoPath,'Uncompressed AVI');
    aviobj.FrameRate=fps;
    open(aviobj);
    for i=1:FramePerVideo
        Index=(VideoNum-1)*FramePerVideo+i;
        im=imread(list{Index});
        writeVideo(aviobj,im);
    end
    close(aviobj);
end

% %% check video is right
% for VideoNum=1:length(list)/FramePerVideo
%     VideoName=['v_ApplyEyeMakeup_g01_c',num2str(VideoNum,'%.3d'),'.avi'];
%     VideoPath=['L:\TmpVideoForDataset\',VideoName];
%     
%     obj = VideoReader(VideoPath); 
%     numFrames = obj.NumberOfFrames;
%     if numFrames==FramePerVideo
%         for i = 1 : numFrames
%             frame = read(obj,i); 
%             Index=(VideoNum-1)*FramePerVideo+i;
%             im=imread(list{Index});
%             flag=psnr(im,frame);
%             if flag~=Inf
%                 flag
%             end
%         end
%     else
%         fprintf('wrong!\n');
%     end
% end