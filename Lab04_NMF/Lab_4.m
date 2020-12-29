% Notes: eps = very small value (you can add it to both B and W, it doesn't
% matter. it just makes sure you dont get zeros)

% 1W' is a vector of column sums

% ones(size(B))

%% Lab 4 - Non-negative Matrix Factorization
%% Submission Instructions
% * Submission Deadline: Thursday (1nd October 2020) at 3:00PM EST.
% * Please submit the lab in any one of these forms (we _will _be strict about 
% students adhering to these formats - 
% 
% 1. A completed |Lab_4.mlx| with the corresponding |nmf.m|, |ssnmf.m|, |compute_objective.m|, 
% |compute_objective_ss.m| files.
% 
% 2. A completed |Lab_4.m| file *and *an exported PDF with the corresponding 
% |.m| files as above. To export your |.m| file to a PDF, use the Publish Tab 
% in the Editor. It is *necessary* to label all the figures you include, if they 
% have not been labelled in the helper code already.
%% The ORL Faces Dataset
% We use the ORL Face database for this assignment, which consists of 400 images 
% for 40 people, each of size $112 \times 92$. These images were taken at different 
% times, with varying lighting and for different facial expressions. All faces 
% are in an upright position with a frontal view, with a slight left-right rotation. 
% To use this dataset, we perform some pre-processing on them, listed here - 
% 
% # We use only the train split for this data (the first 9 images per person).
% # We construct a data matrix |matX| of size $10304 \times 360$ by flattening 
% all the faces. 
% # |matX| is divided by the max value present in the images to normalize the 
% data and avoid overflow issues, giving us the final data matrix |V|.

clear all; close all;
d1 = 112; d2 = 92; d = d1*d2; 
num_images = 9; num_people = 40; 
images = cell(num_people, num_images);
matX = zeros(d, num_people*num_images);

count = 1;
for i = 1:num_people
    for j = 1:num_images
        % filename = sprintf('/your/local/path/to/orl_faces/Train/s%i/%i.pgm', i, j);
        filename = sprintf('/Users/johnduva/MATLAB-Drive/Lab_4/orl_faces/Train/s%i/%i.pgm', i, j);
        img = double(imread(filename));

        matX(:,count) = reshape(img, d, 1);
        count = count+1;
    end
end

V = matX/max(matX(:));

%% Validation on the ORL Faces Dataset
% *Step 1*: Output the new bases and weights and plot Calling your NMF function 
% on the data matrix |X| with parameters |rank = 40|, |max_iter=500|, and |lambda=0.001| 
% will look something like
%%
[B, W, obj, k] = nmf(V, 40, 500, 0.001);
%%
figure;
subtitle('Basis functions obtained by NMF');
for k = 1:40
  subplot(5, 8, k);
  imagesc(reshape(B(:,k), d1, d2));
  colormap gray; axis image off;
end
%% 
% *Step 2*: Compare your results with MATLAB's predefined NMF function 
%%
opt = statset('MaxIter', 500, 'Display', 'final'); 
[B, W] = nnmf(V, 40, 'options', opt, 'algorithm', 'mult');
%%
figure;
subtitle('Basis functions obtained by MATLAB NMF Function');
for k = 1:40
  subplot(5, 8, k);
  imagesc(reshape(B(:,k), d1, d2));
  colormap gray; axis image off;
end
%% Performing NMF with added sparsity constraints
% The process for performing sparse NMF is the same as above, with a few changes 
% to Step 4 (the update rules).
% 
% # Create a sparse NMF function |ssnmf(V, rank, max_iter, lambda, alpha, beta)|
% # Initialize $B$ and $W$randomly, and make sure $W$has unit-sum columns (each 
% column should sum to $1$).
% # Calculate the initial objective. It will be helpful to define a new function 
% for sparse NMF, |compute_objective_ss(V, W, H, alpha, beta)| that returns the 
% objective value.
% # Perform the iterations $B = B \otimes \frac{ \Big( \frac{V}{BW} \Big) W^T}{1 
% W^T + \beta}$ and $W = W \otimes \frac{ B^T \Big( \frac{V}{BW} \Big) }{B^T 1 
% + \alpha}$, where $\otimes$ specifies element-wise multiplication and all divisions 
% are element-wise division.
% # Calculate the new objective function value.
% # Stopping Criteria: Stop when the absolute difference of objective values 
% is smaller than or equal to $\lambda$ (or) the max number of iterations has 
% been reached.
% 
% *Notes - *
% 
% * Boilerplate for |ssnmf(V, rank, max_iter, lambda)| has been provided in 
% |ssnmf.m|
% * Boilerplate for |compute_objective_ss(V, W, H)| has been provided in |compute_objective_ss.m 
% |
% * Here, the notation $1 W^T + \beta$ and $B^T 1 + \alpha$ are normalizing 
% the columns of $W$and $B$ by the sum of column elements plus $\beta$, and the 
% sum of column elements plus $\alpha$ respectively. You should ensure you perform 
% this normalization.
%% Validation on the ORL Faces Dataset
% *Step 1*: As before, output the new bases and weights and plot them by calling 
% your SSNMF function on the data matrix |V| with parameters |rank = 40|, |max_iter=500|, 
% |lambda=0.001|, |alpha=100|, and |beta=1| will look something like
%%
[B, W, obj, k] = ssnmf(V, 40, 500, 0.001, 100, 1);
%%
figure;
subtitle('Basis functions obtained by Sparse NMF');
for k = 1:40
  subplot(5, 8, k);
  imagesc(reshape(B(:,k), d1, d2));
  colormap gray; axis image off;
end