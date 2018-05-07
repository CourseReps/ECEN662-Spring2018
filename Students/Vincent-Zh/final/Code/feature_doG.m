function a = feature_doG(origin)

    row=128;
    colum=128;
    sigma0=sqrt(2);
    img=origin;
    octave=3;
    level=3;
    D=cell(1,octave);
    for i=1:octave
        D(i)=mat2cell(zeros(row*2^(2-i)+2,colum*2^(2-i)+2,level),row*2^(2-i)+2,colum*2^(2-i)+2,level);
    end

    temp_img=kron(img,ones(2));
    temp_img=padarray(temp_img,[1,1],'replicate');

    for i=1:octave
        temp_D=D{i};
        for j=1:level
            scale=sigma0*sqrt(2)^(1/level)^((i-1)*level+j);
            p=(level)*(i-1);

            f=fspecial('gaussian',[1,floor(6*scale)],scale);
            L1=temp_img;
            if(i==1&&j==1)
                L2=conv2(temp_img,f,'same');
                L2=conv2(L2,f','same');
                temp_D(:,:,j)=L2-L1;

                L1=L2;
            else
                L2=conv2(temp_img,f,'same');
                L2=conv2(L2,f','same');
                temp_D(:,:,j)=L2-L1;
                L1=L2;
                if(j==level)
                    temp_img=L1(2:end-1,2:end-1);
                end

                if(i==3&&j==3)
                    a = uint8(255 * mat2gray(temp_D(:,:,j)));
                end
            end
        end
        D{i}=temp_D;
        temp_img=temp_img(1:2:end,1:2:end);
        temp_img=padarray(temp_img,[1,1],'both','replicate');
    end
end