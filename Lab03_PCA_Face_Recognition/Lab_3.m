%% Lab 4: PCA-based Face Recognition
%% Dataset
% We will use the ORL database, available to download on AT&T’s web site. This 
% database contains photographs showing the faces of 40 people. Each one of them 
% was photographed 10 times. These photos are stored as grayscale images with 
% 112x92 pixels. 
% 
% In our example, we construct a catalog called |orlfaces|, comprised of 
% people named $s_1, s_2, . . . , s_{40}$, each one of them containing 10 photographs 
% of the person. The data has already been split into a training and testing split, 
% where for each person, we use the first 9 photographs for training and the last 
% photograph for test.

%% 1. Load the training data 
% 2. Change each $(d_1, d_2) = (112, 92)$ photograph into a vector
% Your code goes here

% I combined steps 1 and 2 :)
cd '/MATLAB Drive/Lab_3/orl_faces/Train/'
big_training_matrix = [];
for folder = 1: length(dir('s*'))
    for file = 1 : 9
        matrix = imread(['s' mat2str(folder) '/' mat2str(file) '.pgm' ]);
        matrix2 = reshape(matrix, [10304 1]);
        big_training_matrix = cat(2, big_training_matrix,  matrix2);
    end
end
clearvars file folder matrix matrix2

%% 
% 3. Using all the training photographs for the $N$ people in the training 
% dataset, construct a subspace $H$with dimensionality less than or equal to $N$such 
% that this subspace has the maximum dispersion for the $N$ projections. To extract 
% this subspace, use Principal Component Analysis, as described below - 
% 
% * Center the data 
avg = mean(big_training_matrix, 2);
centered = double(big_training_matrix) - avg;

% * Compute the correlation matrix 
corr_matrix = corrcoef(centered'); 

% * Use either the |SVD| or |eig| functions to perform SVD and get the eigenvectors 
% and eigenvalues for the correlation matrix.
% V eigenvectors, %S is eigenvalues 
[V,S] = eig(corr_matrix); 

% Eigenvectors already normalized %

%% 
% 4. Plot the eigenvalues
lamdba = diag(S);
plot(lamdba)

%% 
% 5. Plot the first 3 eigenfaces and the last eigenface (these will be the 
% correctly reshaped eigenvectors)
subplot(1,4,1)
imshow(reshape(V(:,end), [112 92]), []); 
subplot(1,4,2)
imshow(reshape(V(:,end-1), [112 92]), []); 
subplot(1,4,3)
imshow(reshape(V(:,end-2), [112 92]), []); 
subplot(1,4,4)
imshow(reshape(V(:,1), [112 92]), []); 

%% 
% 6. Pick a face and reconstruct it using $k = {10, 20, 30, 40}$ eigenvectors. 
% Plot all of these reconstructions and compare them. For each value of $k$, plot 
% the original image, reconstructed image, and the difference b/w the original 
% image and reconstruction in each case. Write your observations.

% select a face
X = double(big_training_matrix(:,end))-avg;
for k = 10 : 10 : 40 
    % get the eigenvectors
    E = V(:,end-k:end);
    % find the w's for the 40 eigenbases
    w = E'*X; 
    C = E*w + avg;
    
    figure(k)
    hold on;
    % original
    subplot(1,3,1)
    imshow(reshape(X, [112 92]),[])
    % reconstructed 
    subplot(1,3,2)
    imshow(reshape(C, [112 92]), [])
    % difference
    subplot(1,3,3)
    imshow(reshape(C-double(big_training_matrix(:,1)), [112 92]), [])
    hold off;
end

% I observed that as we increased k (the number of eigenfaces), we improve
% the accuracy of the reconstruction.


%% 7. Load the testing data, and reshape it similar to the training data.
cd '/MATLAB Drive/Lab_3/orl_faces/Test/'
big_testing_matrix = [];
for folder = 1: length(dir('s*'))
    for file = 10 : 10
        matrix = imread(['s' mat2str(folder) '/' mat2str(file) '.pgm' ]);
        matrix2 = reshape(matrix, [10304 1]);
        big_testing_matrix = cat(2, big_testing_matrix,  matrix2);
    end
end
clearvars file folder matrix matrix2

% 8. For each photograph in the testing dataset, you will implement a classifier 
% to predict the identity of the person. To do this, follow these steps - 

% * Determine the projection of each test photo onto $H$ with different dimensionalities 

% * Compare the distance of this projection to the projections of all images 
% in the training data.

% * For each test photo's projection, find the closest category of projection 
% in the training data.

% each row of 'distList' is one of four dimensionalities (10,20...), each column is one of 40 testing faces
distList = cell(4, size(big_testing_matrix,2));
% each row of 'distList' is one of four dimensionalities, each column is the person with minimum distance
people = cell(4, size(big_testing_matrix,2));
% Using varying dimensionalities)...
for d = 10 : 10 : 40
    % ...compare each of 40 testing faces...
    for photo = 1 : size(big_testing_matrix, 2) 
        % (using the 'd' best eigenvectors...
        E = V(:, end-d:end);
        % ...and the centered testing image...
        X = double(big_testing_matrix(:,photo))-avg;
        % ...and the projection matrix of the test face)...
        w = E' * X;
        proj = E*w + avg;

        % ...to projection of each original face.
        for face = 1 : size(big_training_matrix, 2)
            E2 = V(:, end-d:end);
            X2 = double(big_training_matrix(:,face))-avg;
            w2 = E2' * X2;
            proj2 = E2*w2 + avg;
            
            % get the euclidean norm distance
            dist = norm(proj - proj2);
            % save it to the apporiate cell
            distList{(d/10),photo}(1,face) = dist;
        end
        [MIN, i] = min(distList{1,photo});
        people{(d/10),photo} = [ceil(i/9), i];
    end
end

% 'person' variable:
% each column indicates one of 40 testing faces
% each row refers to the dimensionality
% each cell is a two-element array that indicates (1) which person that
% training face is, and (2) which training photo had the smallest distance from it

%% 
% 9. Show the closest image in the training dataset for the s1 test example.
% Your code goes here
image = people{1,1}(2);
imshow(reshape(big_training_matrix(:,image), [112 92]),[])


