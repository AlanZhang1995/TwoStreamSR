clear;
%% load dataset information
ds_info=importdata('data_info.txt');
video_name=ds_info.textdata;
video_numFrame=ds_info.data(:,1);
%% output file
frame_path='I:\CDVL134\frame\';
fid=fopen('step1_train_image_list.txt','w');
for i=1:length(video_name)
    %% random select 120 frame between frame6 to numFrame-5
    N=video_numFrame(i)-10;
    select_frame=randperm(N);
    select_frame=select_frame(1:120)+5;
    %% write selected frame path to txt
    for j=1:length(select_frame)
%         path=[frame_path video_name{i}(1:end-4) '\img_' num2str(select_frame(j)-1,'%.5d') '.png'];
        path=[frame_path video_name{i}(1:end-4) '\img_' num2str(select_frame(j)-2,'%.5d') '.png'];
        fprintf(fid,'%s\n',path);
        path=[frame_path video_name{i}(1:end-4) '\img_' num2str(select_frame(j),'%.5d') '.png'];
        fprintf(fid,'%s\n',path);
    end
end
fclose(fid);

%% %%%%%%%%%%%%%%%%%%confer_list_is_right%%%%%%%%%%%%%%%%%%%%%%%% 
% %% read train set list
% list=importdata( 'step1_train_image_list.txt');
% %% output file and image fold 
% for i=1:length(list)
%     if exist(list{i},'file')==0
%         list{i}
%     end
% end
%% confer there is no scene switching
% suspects=[];
% suspects_psnr=[];
% for i=1:length(list)
%     if mod(i,2)==1
%         image1=imread(list{i});
%     else
%         image2=imread(list{i});
%         p=psnr(image1,image2);
%         if  p < 18
%             suspects=[suspects,i-1];
%             suspects_psnr=[suspects_psnr,p];
%         end
%     end
% end
% [val,loc]=sort(suspects_psnr);
% j=1;
% for i=j:j+4
%     suspects(loc(i))
%     list{suspects(loc(i))}
%     im=imread(list{suspects(loc(i))});
%     figure;imshow(im);
%     im=imread(list{1+suspects(loc(i))});
%     figure;imshow(im);
%     '----------------------------------------'
% end