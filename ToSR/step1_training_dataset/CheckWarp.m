function flag=CheckWarp(im1,im2,flow)
im2_warp=uint8(WarpInMatlab(im2,flow));
im1_c=im1(11:end-10,11:end-10,:);
im2_c=im2(11:end-10,11:end-10,:);
im2_wc=im2_warp(11:end-10,11:end-10,:);
if psnr(im1_c,im2_c)<psnr(im1_c,im2_wc)
    flag=1;
else
    flag=0;
end