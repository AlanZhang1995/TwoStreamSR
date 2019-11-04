function image1=WarpInMatlab(image2,flow)
[hei,wid,dim]=size(image2);
image1=zeros(hei,wid,dim);
for x=1:wid
    for y=1:hei
        x2=x+flow(y,x,1);
        y2=y+flow(y,x,2);
        if(x2>=1 && y2>=1 && x2<=wid && y2<=hei)
            ix2_L = max(floor(x2),1);
            iy2_T = max(floor(y2),1);
            ix2_R = min(ceil(x2), wid);
            iy2_B = min(ceil(y2), hei);

            alpha=x2-ix2_L;
            beta=y2-iy2_T;

            for c=1:dim
                TL = image2(iy2_T,ix2_L,c);
                TR = image2(iy2_T,ix2_R,c);
                BL = image2(iy2_B,ix2_L,c);
                BR = image2(iy2_B,ix2_R,c);

                image1(y,x,c) = (1-alpha)*(1-beta)*TL + alpha*(1-beta)*TR +(1-alpha)*beta*BL + alpha*beta*BR;
            end
        else
            for c=1:dim
               image1(y,x,c) = 0;
            end
                
        end
    end
end