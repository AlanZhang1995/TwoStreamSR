function result=readMatFlow(filename)
% filename='/home/alan/ApplyEyeMakeup/result/v_ApplyLipstick_g01_c03/flow_x_00001.mat';
fid=fopen(filename,'rb');
[hei,~]=fread(fid,1,'int');
[wid,~]=fread(fid,1,'int');
[type,~]=fread(fid,1,'int');
if type==5
    type_tag='single';
elseif type==6
    type_tag='double';
else
    fprintf('type=%d,please check tag referring https://www.jianshu.com/p/204f292937bb\n',type);
    result=-1;
    return;
end
[mat,COUNT]=fread(fid,type_tag);
if COUNT==hei*wid
    result = reshape(mat,[wid,hei]);%reshape是列主序
    result=result';
else
    fprintf('file wrong!!\n');
    result=-1;
end
fclose(fid);
% result2=result;
% result2(result2>20)=20;
% result2(result2<-20)=-20;
% result2=uint8((result2+20)/40*255);
% figure;imshow(result2);

% load([filename(1:end-4),'.txt']);
% result2=flow_x_00001;
% result2(result2>20)=20;
% result2(result2<-20)=-20;
% result2=(result2+20)/40*255;
% figure;imshow(uint8(result2));