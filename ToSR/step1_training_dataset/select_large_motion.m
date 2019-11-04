function select_point=select_large_motion(im,patch_size,stride_wid,stride_hei,num)
[hei,wid]=size(im);

%random select start point
k_hei=floor((hei-patch_size+1)/stride_hei);
k_wid=floor((wid-patch_size+1)/stride_wid);
hei_bound=hei-patch_size+1-stride_hei*k_hei;
wid_bound=wid-patch_size+1-stride_wid*k_wid;

hei_start=ceil(rand(1)*hei_bound);
wid_start=ceil(rand(1)*wid_bound);

point={[hei_start,wid_start]};
response=sum(sum(im(point{1}(1)+10:point{1}(1)+patch_size-1-10,point{1}(2)+10:point{1}(2)+patch_size-1-10)));
for i=hei_start:stride_hei:hei-patch_size+1
    for j=wid_start:stride_wid:wid-patch_size+1
        tmp=sum(sum(im(i+10:i+patch_size-1-10,j+10:j+patch_size-1-10)));
        response=[response,tmp];
        point=[point,[i,j]];
    end
end
[~,index]=sort(response,'descend');
select_point={};
for i=1:num
    select_point=[select_point,point{index(i)}];
end