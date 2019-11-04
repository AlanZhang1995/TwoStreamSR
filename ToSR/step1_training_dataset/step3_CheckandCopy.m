% % % %Ubuntu
% % % list=importdata( './step1_train_image_list.txt');
% % % off=0;
% % % for i=1:797
% % %     TestFolder = ['/media/alan/Passport/TmpFrameForDataset/v_ApplyEyeMakeup_g01_c',num2str(i,'%.3d'),'/'];
% % %     filepaths = dir([TestFolder,'img*.png']);
% % %     for j=1:length(filepaths)
% % %         index=off+j;
% % %         im1 = imread(fullfile(TestFolder,filepaths(j).name));
% % %         path2=['/media/alan/Passport',strrep(list{index}(3:end),'\','/')];
% % %         im2 = imread(path2);
% % %         flag=psnr(im1,im2);
% % %         if flag~=Inf
% % %             flag
% % %         end
% % %     end
% % %     off=off+length(filepaths);
% % % end
% % % 
% % % fprintf('Check Finished!/n');
% % % 
% % % fid=fopen([pwd,'/step3_flow_tvl1_list.txt'],'w');
% % % index=1;
% % % for i=1:797
% % %     TestFolder = ['/media/alan/Passport/TmpFrameForDataset/v_ApplyEyeMakeup_g01_c',num2str(i,'%.3d'),'/'];
% % %     for j=1:2:39
% % %         src = [TestFolder,'flow_x_',num2str(j,'%.5d'),'.mat'];
% % %         dst = ['/media/alan/Passport/dataset2/flow_tvl1/','flow_x_',num2str(index,'%.5d'),'.mat'];
% % %         movefile(src, dst);
% % % 		fprintf(fid,'%s\n',strrep(dst,'/media/alan/Passport','L:'));
% % %         
% % %         src = [TestFolder,'flow_y_',num2str(j,'%.5d'),'.mat'];
% % %         dst = ['/media/alan/Passport/dataset2/flow_tvl1/','flow_y_',num2str(index,'%.5d'),'.mat'];
% % %         movefile(src, dst);
% % %         fprintf(fid,'%s\n',strrep(dst,'/media/alan/Passport','L:'));
% % % 		
% % %         index=index+1;
% % %     end
% % % end
% % % fclose(fid);

%Windows
% list=importdata( './step1_train_image_list.txt');
% off=0;
% for i=1:797
%     TestFolder = ['I:/TmpFrameForDataset/v_ApplyEyeMakeup_g01_c',num2str(i,'%.3d'),'/'];
%     filepaths = dir([TestFolder,'img*.png']);
%     for j=1:length(filepaths)
%         index=off+j;
%         im1 = imread(fullfile(TestFolder,filepaths(j).name));
%         path2=['I:',list{index}(3:end)];
%         im2 = imread(path2);
%         flag=psnr(im1,im2);
%         if flag~=Inf
%             flag
%         end
%     end
%     off=off+length(filepaths);
% end
% 
% fprintf('Check Finished!/n');

fid=fopen([pwd,'/step3_flow_tvl1_list.txt'],'w');
index=1;
for i=1:797
    TestFolder = ['I:/TmpFrameForDataset/v_ApplyEyeMakeup_g01_c',num2str(i,'%.3d'),'/'];
    for j=1:2:39
        src = [TestFolder,'flow_x_',num2str(j,'%.5d'),'.mat'];
        dst = ['I:\dataset2\flow_tvl1\','flow_x_',num2str(index,'%.5d'),'.mat'];
        movefile(src, dst);
		fprintf(fid,'%s\n',dst);
        
        src = [TestFolder,'flow_y_',num2str(j,'%.5d'),'.mat'];
        dst = ['I:\dataset2\flow_tvl1\','flow_y_',num2str(index,'%.5d'),'.mat'];
        movefile(src, dst);
        fprintf(fid,'%s\n',dst);
		
        index=index+1;
    end
end
fclose(fid);