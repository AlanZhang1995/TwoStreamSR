function sequence=find_sequence(hr_path)
%extract frame number
%construct path
aa=regexp(hr_path, '\', 'split');
path=aa{1};
for i = 2:length(aa)-1
    path=[path,'\',aa{i}];
end
bb=aa{end};
aa=regexp(bb, '_', 'split');
bb=aa{end};
aa=regexp(bb, '\.', 'split');
num=str2num(aa{1});

%load image
im_hr=imread(hr_path);
Size=size(im_hr);
sequence=zeros([Size,7],'uint8');
for i=1:7
    im_path=[path,'\img_',num2str(num+i-4,'%.5d'),'.png'];
    sequence(:,:,:,i)=imread(im_path);
end