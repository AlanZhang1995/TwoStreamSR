%(Total:15940 frame pair or 159400 patches pair)
%(Frame: Train:14896(/16=931) Validation:1044(/1.6=652.5))
%(Patch: Train:148960(/160=931) Validation:10440(/16=652.5))
clear;
number=1044;

list=importdata( 'step1_train_image_list.txt');
list_flow=importdata( 'step3_flow_tvl1_list.txt');

select_patch=randperm(length(list)/2);
select_patch=select_patch(1:number);
select_patch=select_patch*2-1;

fid1=fopen('step4_train_frame1_list.txt','w');
fid2=fopen('step4_train_frame2_list.txt','w');
fid3=fopen('step4_train_flow_list.txt','w');
fid4=fopen('step4_valid_frame1_list.txt','w');
fid5=fopen('step4_valid_frame2_list.txt','w');
fid6=fopen('step4_valid_flow_list.txt','w');

for i = 1 : 2 : length(list_flow)
    if ismember(i,select_patch)
        fprintf(fid6,'%s\n',list_flow{i});
        fprintf(fid6,'%s\n',list_flow{i+1});
        fprintf(fid4,'%s\n',list{i});
        fprintf(fid5,'%s\n',list{i+1});
    else
        fprintf(fid3,'%s\n',list_flow{i});
        fprintf(fid3,'%s\n',list_flow{i+1});
        fprintf(fid1,'%s\n',list{i});
        fprintf(fid2,'%s\n',list{i+1});
    end
end
fclose(fid1);
fclose(fid2);
fclose(fid3);
fclose(fid4);
fclose(fid5);
fclose(fid6);