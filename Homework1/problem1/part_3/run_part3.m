%% Driver File for Problem 1: Part 3: Gender Detection
% You will build a gender detection system using the PCA dimensions 
% from images.
% Author Name : (John D'Uva)

%% Get male faces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/lfw_genders/male/train/'
filenames = dir('*.jpg');
male = [];
for files = 1 : length(filenames)
   male{files,1} = filenames(files).name;
end
flatTrain = (size(imread('Adel_Al-Jubeir_0001.jpg'), 1)^2);
matX = []; 
for i = 1 : length(male)
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/'...
        'data/lfw_genders/male/train/' cell2mat(male(i))];
    img = double(imread(filename));
    matX(:,i) = reshape(img, flatTrain, 1);
end
male = matX/max(matX(:));
% clearvars matX i img filename filenames files names ans

%% Get female faces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/lfw_genders/female/train/'
filenames = dir('*.jpg');
female = [];
for files = 1 : length(filenames)
   female {files,1} = filenames(files).name;
end
flatTrain = (size(imread('Ai_Sugiyama_0001.jpg'), 1)^2);
matX = []; 
for i = 1 : length(female )
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/'...
        'data/lfw_genders/female/train/' cell2mat(female (i))];
    img = double(imread(filename));
    matX(:,i) = reshape(img, flatTrain, 1);
end
female = matX/max(matX(:));
clearvars matX i img filename filenames files names ans

%% Test matrix
% do the same thing to import and reshape test matrices
% Get female faces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/lfw_genders/female/test/'
filenames = dir('*.jpg');
test_female = [];
for files = 1 : length(filenames)
   test_female{files,1} = filenames(files).name;
end
flatTest = (size(imread('AJ_Cook_0001.jpg'), 1)^2);
matX = []; 
for i = 1 : length(test_female)
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/'...
        'data/lfw_genders/female/test/' cell2mat(test_female (i))];
    img = double(imread(filename));
    matX(:,i) = reshape(img, flatTrain, 1);
end
test_female = matX/max(matX(:));
clearvars matX i img filename filenames files names ans


% Get male faces
cd '/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/data/lfw_genders/male/test/'
filenames = dir('*.jpg');
test_male = [];
for files = 1 : length(filenames)
   test_male{files,1} = filenames(files).name;
end
flatTest = (size(imread('Aaron_Pena_0001.jpg'), 1)^2);
matX = []; 
for i = 1 : length(test_male)
    filename = ['/Users/johnduva/Desktop/MLSP/Homework 1/hw1_template/problem_1/'...
        'data/lfw_genders/male/test/' cell2mat(test_male(i))];
    img = double(imread(filename));
    matX(:,i) = reshape(img, flatTrain, 1);
end
test_male = matX/max(matX(:));
clearvars matX i img filename filenames files names ans

%% Concatenate training faces
all_train = [female male];
% all_train = imresize(all_train, .25);
all_test = [test_female test_male];
% all_test = imresize(all_test, .25);

%% PCA
% Center the training data
centered = all_train - mean(all_train, 'all');
% imshow(reshape( centered(:,1), [250 250]), [])

% Get eigenfaces 'U' and eigenvals 'S' for max K=300
[U,S,V] = svd(centered, 300);
% imshow(reshape( U(:,1), [250 250]), [])

%% 
K_cells = {};
count = 1;
for K = [50, 100, 200, 300]
    % take the K-best eigenvectorsfrom U
    eigFaces = U(:,1:K);
    % save it to the cell matrix
    K_cells{1,count} = eigFaces;
    
    % take the K-best eigenvalues from U
    eigVals = diag(S); eigVals = eigVals(1:K);
    % save it to the cell matrix
    K_cells{2,count} = eigVals;
    % index of for loop 
    count = count+1; 
end

% plot all four eigVal arrays of size K
for fig = 1 : 4
    subplot(2,2,fig)
    plot(K_cells{2,fig})
    ylim([0 500])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The first 50 eigenvectors capture the most info on a "per eigenvector" basis
% but K=300 captures more *total* information.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Average male and female face

avg_male = [];
avg_female = [];
for row = 1 : 62500
    avg_male   = [avg_male mean(male(row,:))];
    avg_female = [avg_female mean(female(row,:))];
end

subplot(1,2,1)
imshow( reshape( avg_male, [250 250]), [])
subplot(1,2,2)
imshow( reshape( avg_female, [250 250]), [])

%% Project Average Male and Female onto 'K' PCA Spaces
% 'avgs':
%    - First row is male avgs, second is female
%    - Each column is a different K
% 'avg_weights':
%    - First row is male avg weights, second is female avg weights
%    - Each column is a different K

avgs = {};
avg_weights = {};
count = 1;
for K = [50, 100, 200, 300]
    E = U(:,1:K);
    X = reshape(avg_male, [62500 1]);
    w = E' * X;
    male_proj = E*w;
    avgs{1,count} = male_proj;
    avg_weights{1,count} = w;
    
    E = U(:,1:K);
    X = reshape(avg_female, [62500 1]);
    w = E' * X;
    female_proj = E*w;
    avgs{2,count} = female_proj;
    avg_weights{2,count} = w;
    
    count = count + 1;
end

%% Project Testing Images onto 'K' PCA Spaces
all_test = [test_male, test_female]; % just run this once during debugging

testing = {}; % size = 4x2000: row = K, column = face
count1 = 1;
% weights of each testing image
testWeights = {};
for K = [50, 100, 200, 300]
    disp(K);
    E = U(:,1:K);
    count2 = 1;
    for face = 1 : size(all_test, 2)
        X = all_test(:,face);
        w = E' * X;
        proj = E*w;
        testing{count1,count2} = proj;
        testWeights{count1,count2} = w;
        count2 = count2 + 1;
    end
    count1 = count1 + 1;
end
% imshow( reshape( testing{2,1}, [250 250]), [])

%% Classify Each Test Image as Male or Female...
% by finding the distance between the weights of each testing image and the
% weights of your average male face and average female face.

% create empty binary array: 1s will be men, 0s will be women
testingDecision = ones(size(testing));

for K = 1 : 4
    % get avg male weight & avg female weight for K
    avgMaleWeight  = avg_weights{1,K}; 
    avgFemaleWeight  = avg_weights{2,K}; 
    
    % compare test weight to 
    for testWeight = 1 : length(testing)
        % get current test weight
        test_weight = weights{K,testWeight};
        
        % get the euclidean norm distance
        maleDist   = norm(test_weight  - avgMaleWeight);
        femaleDist = norm(test_weight  - avgFemaleWeight);

        if maleDist < femaleDist
            testingDecision(K,testWeight) = 1; % image is male
        else
            testingDecision(K,testWeight) = 0; % female
        end
        
    end    
end


%% Calculate Accuracy of Classification Across Testing Data 
% for different values of k ∈ {50,100,200,300} used for projection. Briefly 
% explain the time vs. accuracy trade-off for different values of k that you see.
clc
K2 = [50, 100, 200, 300];
for K = 1:4
    tic
    correctMaleRate = sum(testingDecision(K,1:1000) == 1) / 1000;
    correctFemaleRate = sum(testingDecision(K,1000:end) == 0) / 1000;
    disp("K=" + K2(K) + ":");
    disp(" -correct male accuracy of " + correctMaleRate);
    disp(" -correct female accuracy of " + correctFemaleRate);
    toc
    disp(" ")
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% When it comes to males, there is no benefit to using any K above 50 since
% the accuracy rate remains 52.1%. Similarly with females, there is only a
% minor improvement (60.1% to 60.4%) between K=50 and K=300. The time trade
% of is negligible when computations are performed with the weights.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Develop a gender detection algorithm based on all training images,

% i. Project all the training images into the 'K' PCA dimensions, 
%    and all the testing images into the 'K' PCA dimensions as well.

% Output matrix of training projections for each image, each K
training = {}; % size = 4x3868: row = K, column = trainingFace
trainWeights = {}; % weights of each testing image
count1 = 1;
for K = [50, 100, 200, 300]
    E = U(:,1:K);
    count2 = 1;
    for trainFace = 1:size(all_train,2)
        X = all_train(:,face);
        w = E' * X;
        proj = E*w;
        training{count1,count2} = proj;
        trainWeights{count1,count2} = w;
        count2 = count2 + 1;
    end
    count1 = count1 + 1;
end
% sanity check: imshow(reshape( training{4,1}, [250 250]), [])
% The equivalent 'testing' matrix is already built in lines 168:190

%% ii. Calculate average Euclidean distance 
% between the weights of each test image and weights of all male training  
% images and all female training images.

test50  = cell2mat(testWeights(1, :));
train50 = cell2mat(trainWeights(1,:)) ;

dist_male = [];
dist_female = [];
for testWeight = 1 : size(testWeights,2) % 2000
   dist_male(testWeight) = norm( train50(:,1:1934) - test50(:,testWeight), 1);
   dist_female(testWeight) = norm( train50(:,1935:end) - test50(:,testWeight), 1) ; 
end

% Classify the test image based on which average Euclidean distance is smaller.
decision = []; % males=1, females=0
for dist = 1 : length(dist_male)
    if dist_male > dist_female
        decision(dist) = 0;
    else
        decision(dist) = 1;
    end
end

maleAccuracy = sum(decision(1:1000)==1)/1000;
femaleAccuracy = sum(decision(1000:2000)==0)/1000;

