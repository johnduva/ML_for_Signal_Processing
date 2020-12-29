%% Version 1
% Load the image '5.1.09.tiff' which is a surface in the moon. Assign this image to the 
% the variable Moon_image.
Moon_image = imread('5.1.09.tiff');

% Show Moon_image 
figure(1)
imshow(Moon_image);
 
% Similarly, load and show the image ‘Cameraman.tif’ which is a cameraman. 
% Assign this image to the variable Cameraman_image.
Cameraman_image = imread('Cameraman.tif');
figure(2)
imshow(Cameraman_image);

% Fade for example Moon_image by a factor of 0.8 and Cameraman_image by factor 
% of 0.2 and sum the two matrices in a new matrix named Mixte_image.
Mixte_image = Moon_image*.8 + Cameraman_image*.2;
% Show the Mixte_image 
figure(3) 
imshow(Mixte_image);

% Create a new image named First_part_image_1 by selecting the first 100 components
% of each dimension of image Cameraman_image.
First_part_image_1 = Cameraman_image(1:100,1:100);

% Create a new image named last_part_image_2 by selecting the last 100 components 
% of each dimension of image Moon_image.
last_part_image_2 = Moon_image(end-99:end, end-99:end);

% Fade both images last_part_image_2 by a factor of 0.8 and First_part_image_1 by factor 
% of 0.2 and sum the two matrices in new matrix named last_part_Mixte_image.
last_part_Mixte_image = last_part_image_2*.8 + First_part_image_1*.2;


%% Version 2:
% We will do the same fade and mixing process as version 1 question but this 
% time using matrix and vector multiplication. The first step will consist 
% of changing both image matrices from size of 256x256 each to vector of size
% 65536x1; (To create a vector from matrix uses the function reshape. You 
% can use help to show you how to use reshape function). 
% Create a new matrix  (named Both_images) by appending both image vectors 
% to form a matrix of size 65536x2.
Both_images = [reshape(Moon_image, [65536 1]), reshape(Cameraman_image, [65536 1])];

% Create of vector named Fade_vector of size 2x1 and containing the fade factor 
% values (0.5,0.5) for both image.
Fade_vector = [.5; .5];

% Multiply Both_images matrix and Fade_vector to obtained the mixing_image_vector
mixing_image_vector = im2double(Both_images) * Fade_vector;

% Resize using the function reshape again the obtained vector mixing_image_vector to create
% mixing_image_matrix of size (256x256).
mixing_image_matrix = reshape(mixing_image_vector, [256 256]);
imshow(mixing_image_matrix)

%% Version 3: Image Transition
% Moon_image should fade out linearly and Cameraman_image should fade in linearly.
Fade_vector2 = [...
    1 .9 .8 .7 .6 .5 .4 .3 .2 .1 0;...
    0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1 ...
    ];
% begin recording video
writerObj = VideoWriter('transition.avi');
writerObj.FrameRate = 2;
open(writerObj);

% write the frames to the video
for fade = 1 : length(Fade_vector2)
    mixing_image_vector = im2double(Both_images) * Fade_vector2(:, fade);
    mixing_image_matrix = reshape(mixing_image_vector, [256 256]);
    writeVideo(writerObj, mixing_image_matrix);
    disp(Fade_vector2(:, fade));
end
close(writerObj);