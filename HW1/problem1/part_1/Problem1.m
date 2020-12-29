%% Problem 1: Face Detection
%% A and B: Reading images and building a matrix
% Get vector of filenames from cwd...
clear 
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/lfw_1000'
filenames = dir('*.pgm');
names = [];
for files = 1 : length(filenames)
   names{files,1} = filenames(files).name;
end

flat = 64*64; 
matX = []; %zeros(flat, length(filenames));
% Create 'Y' (each column as a flattened image)
for i = 1 : length(names)
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/lfw_1000/' cell2mat(names(i))];
    img = double(imread(filename));
    matX(:,i) = reshape(img, flat, 1);
end

training_matrix = matX/max(matX(:));
clearvars matX i img filename filenames files names

%% C. Computing eigenfaces
% subtract the mean
centered = training_matrix - mean(training_matrix, 'all');
corrmatrix = centered * centered';

% Get the eigenvectors 
[eigvecs, eigvals] = eig(corrmatrix);
% get the first eigenface (last column is primary)
E = reshape(eigvecs(:,4096), [64 64]);
E2 = mat2gray(E); % make values positive

% Use svd instead to increase speed
[U,S,V] = svd(centered,0);
% U is eigenvectors, S is eigenvalues (primary vec is first column)
% imshow(reshape(mat2gray(U(:,1)), 64, 64)) 

%% D. Scanning Image or Pattern
% Beatles group image 
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/group_photos'
I = imread('Beatles.jpg'); 
I_grey = squeeze(mean(I,3));

[P,Q] = size(I, [1 2]);
[N,M] = size(E, [1 2]);

for i = 1:(P-N) % number of spaces the patch will move down
    for j = 1:(Q-M) % number of spaces the patch will move right
        patch = I_grey(i:(i+N-1), j:(j+M-1));
        m(i,j) = E2(:)' * double(patch(:)) / norm(double(patch(:)));
%         tmpim = conv2(patch, fliplr(flipud(E2)));
%         convolvedimage = tmpim(N:end, M:end);
%         sumE2 = sum(E2(:));
%         patchscore = convolvedimage - sumE2 * mean(patch(:));
    end
end

%% E. Same thing with factoring of the group image
factor = [0.5, 0.75, 1, 1.5, 2.0];
% Each of four cells will be the matrix of scores (using each of the four factors)
scaled = {};
count = 1; % index of each factor
numPatches = 40;
for f = factor
    tic
    % the scaled size of Beatles image dimensions (P=x, Q=y)
    Pnew = P*f; Qnew = Q*f;
    % rescale Beatles image 
    I_scaled = imresize(I_grey,[Pnew,Qnew]);
    scaled{2,count} = []; scaled{3,count} = []; scaled{4,count} = [];
    
    for i = 1:(Pnew-N) % number of spaces the patch will move down
        for j = 1:(Qnew-M) % number of spaces the patch will move right
            % get the patch itself from the group image
            patch = I_scaled(i:(i+N-1), j:(j+M-1));
            % get the score of the patch
            scaled{1,count}(i,j) = E2(:)' * double(patch(:)) / norm(double(patch(:)));
            
            % if length(array)<10 
            if length(scaled{2,count}) < numPatches
                % put the new score in 2nd row
                scaled{2,count} = [scaled{2,count} scaled{1,count}(i,j)];
                % append the flattened patch to the end of the matrix     
                scaled{3,count} = [scaled{3,count} reshape(patch, 4096, 1)];
                scaled{4,count} = [scaled{4,count}; [i j] ];
                
            % if current score is greater than everything in array, and length(array)==10 
            elseif ismember(0, scaled{1,count}(i,j) >= scaled{2,count}) && length(scaled{2,count}) == numPatches
                % then replace the smallest elem in the array
                [MIN, IDX] = min(scaled{2,count});
                scaled{2,count}(IDX) = scaled{1,count}(i,j);
                % replace the min column with the flattened patch
                scaled{3,count}(:,IDX) = reshape(patch, 4096, 1);
                % replace the min coordinates
                scaled{4,count}(IDX,:) = [i j];
            
            % if length(array)<10 && current score is lower than everything in array
            else
                continue
            
            end 
        end 
    end
    disp(f + " scalar complete.")
    count = count + 1;
    toc
    disp(" ")
end

%% Display the top 40 faces from first factor
%.5 factor
factor = 2;
for face = 1 : length(scaled{2,1})
    subplot(8,5,face)
    hold on
    title(face)
    imshow(reshape(scaled{3,factor}(:,face), [64 64]), [])
    hold off
end

%% Show the 4 faces
figure(2)
hold on
% subplot(1,4,1)
% imshow(reshape(scaled{3,1}(:,35), [64 64]), [])
imshow(reshape(scaled{3,1}(:,35), [64 64]), [], ...
    'XData', [scaled{4,1}(35,1)*2 scaled{4,1}(35,1)*2+64],...
    'YData',[scaled{4,1}(35,2)*2 scaled{4,1}(35,2)*2+64])
% axes('Position', [0, 0, 0, 0]);
imshow(I)
hold off

subplot(1,4,2)
imshow(reshape(scaled{3,1}(:,23), [64 64]), [])

subplot(1,4,3)
imshow(reshape(scaled{3,2}(:,40), [64 64]), [])

subplot(1,4,4)
imshow(reshape(scaled{3,1}(:,28), [64 64]), [])
