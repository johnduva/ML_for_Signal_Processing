% Lab 9: Generating Images Using Proabilistic PCA
% We will also use the transformation matrix obtained
% by this method to generate never before seen samples 
% of handwritten digits using the MNIST Dataset.

trainImgs = loadMNISTImages('train-images-idx3-ubyte');  % training set images (9912422 bytes)
trainLbls = loadMNISTLabels('train-labels-idx1-ubyte');  % training set labels (28881 bytes)

testImgs = loadMNISTImages('t10k-images-idx3-ubyte');    % test set images (1648877 bytes)
testLbls = loadMNISTLabels('t10k-labels-idx1-ubyte');   % test set labels (4542 bytes)

% 1. Define X(i) as all training images that belong to class i = 0, 1, . . . , 9. 
% For example, all images belonging to class 0 would result in X(0) ∈ R784×5923, as there are 5923 training images of digit 0. 		 

% Each column represents 0-9, each cell contains indices of images that correspond to that digit
arr = cell(1,10); 
for i = 0:9
    arr{i+1} = find(trainLbls==i); % find indices of images with label 'i'
    X = trainImgs(:, arr{1,i+1}); % X(i) = all training images of class i = 0, 1, ... , 9
    avg = mean(X,2); % get the mean for each pixel (NOT within class mean) 
    D = size(X, 1);
    N = size(X, 2);
    X = X - avg;
    for K = 50 %[50, 100]
        W = rand(D, K);
        sigma_sq = rand();
        I = eye(K);        
        count = 1;
        % Update W and sigma_sq:
        for iter = 1 : 30  % usually set to 1e6 if using convergence threshold, but takes too long
            M = W'*W + sigma_sq*I;
            U = chol(M);
            V = inv(U);
            M_inv = V*V';
            T = inv(U')*(W'*X); %#ok<MINV> 
            log_M = 2*sum(log(diag(U))) + (D-K)*log(sigma_sq);
            S = 0;
            for col = 1: size(X, 2) %5923
                S = S + norm(X(:,col))^2;
            end
            Tr_SM = (S - sumsqr(T)) / (N*sigma_sq);
            
            % Update log-likelihood
            if exist('LogLik_aprx', 'var')
               LogLik_old = LogLik_aprx;
            else 
                LogLik_old = 0;
            end
            LogLik_aprx = -(N/2)*( D*log(2*pi) + log_M + Tr_SM );
            
            % Use E-step to obtain mean and variance of the hidden variables, zn
            E_zn = zeros(K, size(X, 2) );
            % instantiate M-step terms:
            first = 0; 
            second = 0;
            for col = 1: size(X, 2)
                % E-step terms for M-step:
                E_zn(:,col) = M_inv * W' * X(:,col);
                E_zn_znT = (sigma_sq * M_inv) + (E_zn(:,col) * E_zn(:,col)');
                % M-step terms for W_new:
                first  = first  + X(:,col)*E_zn(:,col)';
                second = second + E_zn_znT;
            end
            W_new = first * inv(second);
            
            % Update 'sigma_sq':
            term = 0;
            for col = 1: size(X, 2)
                E_zn(:,col) = M_inv * W' * X(:,col);
                E_zn_znT = (sigma_sq * M_inv) + (E_zn(:,col) * E_zn(:,col)');
                term = term + (norm(X(:,col))^2 - 2*E_zn(:,col)'*W_new'*X(:,col) + trace(E_zn_znT*W_new'*W_new));
            end
            sigma_sq = (1/(N*D) * term)^2;
            
            % Not currently using this, takes too long but is more accurate:
            % Check if log-likelihood has converged yet:
%             count = count + 1;
%             if abs(LogLik_old - LogLik_aprx) < 1
%                 break
%             end 
        end
        
        % Sample 25 points from k-dimensional Gaussian distribution, z,
        % and multiply by 'W_new' to generate 25 new images from class i:
        z = randn(K);
        z_gen = z(:, 1:25);
        x_gen = (W_new * z_gen) + avg;
        % sanity check: imshow(reshape(x_gen(:,10), [28 28]), [])
        
        % Plot images 
        figure(i+1)
        for plots = 1 : 25
            subplot(5,5,plots)
            imshow(reshape(x_gen(:,plots), [28 28]))
        end
        drawnow

    end
    disp("Completed digit: " + num2str(i));
end
% clearvars ans X W I M S K N D T U V i j Tr_SM M_inv LogLik_aprx log_M 