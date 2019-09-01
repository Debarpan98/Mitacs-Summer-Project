% Created 2019/06/14
% Author: Fatemeh Zabihollay


clear all;
close all;

% Read original WSI and visualize it


files = dir ('./data/py_wsi/data/images/*.svs');

for file = files'
    
    dirname = file.name;
    dirname = split(dirname, '_');
    dirname = dirname(3);
    dirname = split(dirname,'.');
   
    dirname = string(dirname(1));
    dirpath = './data/py_wsi/db/images/';
    path = strcat(dirpath, strcat(dirname,'/'));
    mkdir (path)
    
    orig_wsi = imread(strcat('./data/py_wsi/data/images/',file.name),'Index',3);
    %imshow(orig_wsi);

    patch_dim = 256;
    ID = 0;

    %cd 'data/py_wsi/db/images'


    [m,n,l] = size(orig_wsi); 
    for i= 1 :  1: mod(m, patch_dim )
        %if (patch_dim + i -1 > x1)
        %    break
        %end
        for j = 1 : 1 : mod(n, patch_dim)
            %if (patch_dim + j-1 > y1)
            %    break
            %end
            img_patch = orig_wsi(i:(patch_dim + i-1),j:(patch_dim + j-1),:);
            [a, b, c] = size(img_patch);
            if (a == patch_dim && b == patch_dim)
                name = split(file.name, '.');
                name = name(1);
                name = split(name, '_');
                name = name(3);
                name=convertStringsToChars(strcat('img_',name,'_', string(ID),".jpg"));
                %saveas(imagesc(img_patch),name);
                
                imwrite(img_patch, strcat(path,name));
                %imwrite(mask_patch, 'img_0083_%d.jpg',ID);
                ID = ID + 1;
            end 
        end
    end
    break
end


%{

% See the image information
info=imfinfo('01_01_0083.svs');

% Original annotation information
DOMnode = xmlread('01_01_0083.xml');

% Read whole tumor mask binary image and visualize it
t1 = Tiff('01_01_0083_whole.tif','r');
t1_resize = imresize(t1,0.25);
mask_whole = read(t1);
figure();
imshow(mask_whole*255);


% Read viable tumor mask binary image and visualize it
t2 = Tiff('01_01_0083_viable.tif','r');
mask_viable = read(t2);
figure();
imshow(mask_viable*255);
%}