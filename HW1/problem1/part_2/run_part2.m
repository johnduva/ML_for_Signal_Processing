
% For Part 2, you will downsample the data by resizing the image itself and
% gettinge eigenvectors again

%% Driver File for Problem 1: Part 2: Face Detection
% You will implement an Adaboost Classifier to classify between face images
% and non-face images.
% Author Name : (John D'Uva)

%% Your Driver Script Starts Here
% You can use as many auxilliary scripts as you want
% As long as we can run this script to get all the plots and classification
% accuracies we require from Part 2

%% Reading images and building a matrix
% Get faces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/boosting_data/train/face'
filenames = dir('*.pgm');
trainFaces = [];
for files = 1 : length(filenames)
   trainFaces {files,1} = filenames(files).name;
end
flat64 = 64*64; 
matX = []; %zeros(flat, length(filenames));
% Create 'Y' (each column as a flattened image)
for i = 1 : length(trainFaces )
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/boosting_data/train/face/' cell2mat(trainFaces(i))];
    img = double(imread(filename));
    scaledimg = imresize(img,[64 64]);
    matX(:,i) = reshape(scaledimg, flat64, 1);
end
trainFaces  = matX/max(matX(:));
clearvars matX img i files filename filenames ans scaledimg


% Get non-faces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/boosting_data/train/non-face'
filenames = dir('*.pgm');
trainNonFaces = [];
for files = 1 : length(filenames)
   trainNonFaces{files,1} = filenames(files).name;
end
matX = [];
for i = 1 : length(trainNonFaces)
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/boosting_data/train/non-face/' cell2mat(trainNonFaces(i))];
    img = double(imread(filename));
    scaledimg = imresize(img,[64 64]);
    matX(:,i) = reshape(scaledimg, flat64, 1);
end
trainNonFaces = matX/max(matX(:));
clearvars matX img i files filename filenames ans scaledimg


%% Get lfw faces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/lfw_1000'
filenames = dir('*.pgm');
lfwfaces = [];
for files = 1 : length(filenames)
   lfwfaces{files,1} = filenames(files).name;
end
flat64 = 64*64; 
matX = []; 
for i = 1 : length(lfwfaces)
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/lfw_1000/' cell2mat(lfwfaces(i))];
    img = double(imread(filename));
    matX(:,i) = reshape(img, flat64, 1);
end
lfwfaces = matX/max(matX(:));
clearvars matX i img filename filenames files names


%% Get testing faces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/boosting_data/test/face'
filenames = dir('*.pgm');
testFaces = [];
for files = 1 : length(filenames)
   testFaces{files,1} = filenames(files).name;
end
matX = []; 
for i = 1 : length(testFaces)
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/boosting_data/test/face/' cell2mat(testFaces(i))];
    img = double(imread(filename));
    matX(:,i) = reshape(img, flat64, 1);
end
testFaces = matX/max(matX(:));
clearvars matX img i files filename filenames ans


% Get testing nonFaces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/boosting_data/test/non-face'
filenames = dir('*.pgm');
testNonfaces = [];
for files = 1 : length(filenames)
   testNonfaces{files,1} = filenames(files).name;
end
matX = []; 
for i = 1 : length(testNonfaces)
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/boosting_data/test/non-face/' cell2mat(testNonfaces(i))];
    img = double(imread(filename));
    matX(:,i) = reshape(img, flat64, 1);
end
testNonfaces = matX/max(matX(:));
clearvars matX img i files filename filenames ans

%% Learn the K-best 'eigFaces' of lfw dataset, 
% subtract the mean
centered_lfw = lfwfaces - mean(lfwfaces, 'all');
corrmatrix = centered_lfw * centered_lfw';

K = 10;
[eigvecs, eigvals] = eig(corrmatrix);
eigFaces = mat2gray(eigvecs(:,end-(K-1):end)); 
clearvars corrmatrix eigvals 

%% Represent 'trainFaces' as Linear Combos and Weights
avg = mean(trainFaces, 'all');
% for each of 2429 'trainFaces'...
F_faces = []; % size will be 4096x2429 (you're reconstructing 'trainFaces')
faceWeights = [];
 for photo = 1 : size(trainFaces, 2) 
    % get current 'trainFace', subtract mean...
    X = trainFaces(:,photo)-avg;
    % get the K weights that explain most of the variance
    w = eigFaces' * X;
    % calculate the reconstruction/projection
    proj = eigFaces * w + avg; % sanity check: imshow(reshape(proj, [64 64]), []) 
    % append linear combination to F (each column = flattened/reconstructed trainFace)
    F_faces = [F_faces proj];
    % Also represent as K weights per 'trainFace'
    faceWeights = [faceWeights w];
 end
clearvars w X proj eigvals eigvecs photo avg ans

%% Represent 'trainNonFaces' as Linear Combos and Weights
avg = mean(trainNonFaces, 'all');
F_nonFaces = []; nonFaceWeights = [];
% for each of 4548 'trainNonFaces'...
for photo =  1 : size(trainNonFaces, 2) 
    X = trainNonFaces(:,photo)-avg;
    % get the K weights that explain most of the variance
    w = eigFaces' * X;
    % calculate the reconstruction/projection
    proj = eigFaces * w + avg; % sanity check: imshow(reshape(proj, [64 64]), []) 
    % append linear combination to F (each column = flattened/reconstructed trainFace)
    F_nonFaces  = [F_nonFaces proj];
    nonFaceWeights = [nonFaceWeights w];
end
clearvars w X proj eigvals eigvecs photo avg ans

%% Adaboost Classifier to Classify Between Face and nonFace Images


    














%%
    % ...to projection of each original face.
    for face = 1 : size(big_training_matrix, 2)
        E2 = V(:, end-K:end);
        X2 = double(big_training_matrix(:,face))-avg;
        w2 = E2' * X2;
        proj2 = E2*w2 + avg;
        % imshow(reshape(proj, [64 64]), []) % sanity check

        % get the euclidean norm distance
        dist = norm(proj - proj2);
        % save it to the apporiate cell
        distList{(K/10),photo}(1,face) = dist;
    end
    [MIN, i] = min(distList{1,photo});
    people{(K/10),photo} = [ceil(i/9), i];
