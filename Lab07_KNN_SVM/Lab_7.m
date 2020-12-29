%% Lab 7: MNIST Digit Classification with kNN and SVMs

%% Dataset
% The MNIST Dataset is a set of black and white photos of handwritten
% digits with their corresponding labels from 0 to 9.

% Load the files
train_imgs =   loadMNISTImages('/Users/johnduva/Git/MLSP/Lab7_data/train-images-idx3-ubyte.dms');
train_labels = loadMNISTLabels('/Users/johnduva/Git/MLSP/Lab7_data/train-labels-idx1-ubyte.dms');

t10k_imgs =    loadMNISTImages('/Users/johnduva/Git/MLSP/Lab7_data/t10k-images-idx3-ubyte.dms');
t10k_labels =  loadMNISTLabels('/Users/johnduva/Git/MLSP/Lab7_data/t10k-labels-idx1-ubyte.dms');
clearvars ans filename fp magic 

% Show an example image
imshow(reshape(train_imgs(:,10), [28 28]))
%% Problem Statement
% The goal of this lab is to perform classification on the MNIST dataset,
% where the input is an image of a digit and your prediction is a number
% between 0 to 9.
% 
% The basic method for this lab is as follows -
% 
%%  Train and find the PCA dimensions for the training data.
centered = train_imgs - mean(train_imgs, 'all');
corrmatrix = centered * centered';

% Search K to optimize
accuracies = [];
for K = 2 : 20
    disp(K)
    % Get the eigenvectors 
    [eigvecs, eigvals] = eig(corrmatrix);
    % Get K best eigenvectors
    eigNumbers = mat2gray(eigvecs(:,end-(K-1):end)); 
    % sanity check: imshow(reshape(eigNumbers(:,2), [28 28]), []) 

    % # Project the training dataset using the PCA Bases
    % Represent 'train_imgs' as Linear Combos and Weights
    avg = mean(train_imgs, 'all');
    train_projections = zeros(784,60000); % pre-allocate for speed
    train_weights = zeros(K,60000);
     for photo = 1 : size(train_imgs, 2) 
        % get current 'train_img', subtract mean...
        X = train_imgs(:,photo)-avg;
        % get the K weights that explain most of the variance
        w = eigNumbers' * X;
        % calculate the reconstruction/projection
        proj = eigNumbers * w + avg; % sanity check: imshow(reshape(proj, [28 28]), []) 

        % append linear combination to new matrix (each column = flattened/reconstructed train number)
        train_projections(:,photo) = proj;
        % Also represent as K weights per 'train_number'
        train_weights(:,photo) = w;
     end

    % Project Test Image Datasets Using the PCA Bases
    % Represent 't10k_imgs' as Linear Combos and Weights
    test_projections = zeros(784,10000); % pre-allocate for speed
    test_weights = zeros(K,10000);
     for photo = 1 : size(t10k_imgs, 2) % for each of 10k 'test_imgs'...
        X = t10k_imgs(:,photo)-avg;
        w = eigNumbers' * X;
        proj = eigNumbers * w + avg; 
        test_projections(:,photo) = proj;
        test_weights(:,photo) = w;
     end
    % sanity check: imshow(reshape(test_projections(:,1), [28 28]), []) 

    % # Explore the performance of KNN & SVMs by varying number of PCA bases being projected on.
    % 1. KNN
    % Implement the K-nearest neighbors algorithm for the 10 classes that you
    % have. Classify your testing data based on each test image's distance to 
    % its nearest training images (neighbors). Report your result for the 
    % different number of PCA bases being projected on.

    % Preallocate array of K-lowest distances from current 'test_img'
    predictions = NaN(1,length(t10k_imgs));
    % for every test projection...
    for i = 1 : 800 %length(t10k_imgs)
        arr = zeros(1,length(train_projections));
        currTest = t10k_imgs(:,i);

        for j = 1 : length(train_imgs)
            % check distance of current test projection to all train neighbors
            arr(1,j) = norm( currTest - train_imgs(:,j)); 
        end

        % get the 10 smallest distances
        [~,idx] = sort(arr); 
        arr = idx(1:K);
        % see which label that idx correlates to & which num got highest votes 
        predictions(i) = mode(train_labels(arr));
    end

    % Then compare for accuracy with t10K_labels
    count = 0;
    for i = 1 : sum(~isnan(predictions))
       if t10k_labels(i) ==  predictions(i)
           count = count + 1;
       end
    end
    accuracies(K-1) = count / sum(~isnan(predictions));
end
[MAX, IDX] = max(accuracies);
disp("K = " + (IDX+1) + " is the optimal number of nearest neighbors (accuracy = " + MAX);

%% 2. SVM
% Use the MATLAB |svm| function to do 10-class classification of the MNIST
% Data: https://www.mathworks.com/help/stats/support-vector-machine-classification.html>.

% Play with different C values and kernel functions and see how
% they influence your result. Report your best accuracy and settings
% include the dimension, C value, and kernel function you used.
% 
% Submit all your code and a *writeup as PDF* reporting the parameters that
% gave you your best performance. *Do not include the data directory*.
% 
X = train_projections';
Y = train_labels;
% models = {};

% polynomial, linear, gauss
funcs = 'gaussian';
% create a model for each kernel/C combination
count = 9;
for i = 1:3
    for j = [.1 1 10 100]
        t = templateSVM(...
            'KernelFunction', funcs(i), ...
            'BoxConstraint', j,...
            'IterationLimit', 10);  
        Mdl = fitcecoc(X,Y, 'Learners', t,...
            'FitPosterior','off',...
            'Verbose',2);

        % Compute the resubstitution classification error.
        error = resubLoss(Mdl);
        % The classification error on the training data is small, but the classifier 
        % might be an overfitted model. You can cross-validate the classifier using 
        % crossval and compute the cross-validation classification error instead.
% 
%         % Cross-validate Mdl using 10-fold cross-validation.
%         CVMdl = crossval(Mdl); 
%         % CVMdl is a ClassificationPartitionedECOC cross-validated ECOC classifier.
% 
%         % Estimate the generalized classification error.
%         genError = kfoldLoss(CVMdl);
%         % The generalized classification error is 4, which indicates that the ECOC 
%         % classifier generalizes fairly well.

        models{1,count} = Mdl;
        models{2,count} = error;
%         models{3,count} = genError;

        count = count + 1;
    end
end

% Find the accuracy and idx of the model with highest accuracy
[MAX, IDX] = max(cell2mat(models(2,:)));


