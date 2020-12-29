%% Driver Script for HW3, Problem 1
% GMMs for Speaker Recognition
% Auther Name: John D'Uva

%% Import the training data into 20 Cells 
data = cell(20,2);
folder = dir('data/train/');
for files = 1 : length(folder)
    if folder(files).bytes < 1 % if its not real data, skip
        continue
    else
        % fill up 'data' matrix with data and filenames
        data{files-2, 1} = importdata(['data/train/',folder(files).name]);
        data{files-2, 2} = folder(files).name;
    end
end

%% Estimate GMM params for each speaker (i.e. build 10 GMM models)
K = 64; 
num_speakers = 10;
n_epochs = 20; 
speakerModels = cell(3,num_speakers); % will contain final output parameters
likelihoods = zeros(num_speakers,n_epochs); % for plotting LL over training iterations 
for speaker = 1 : num_speakers 
    %% Step 0: Data
    X = [data{speaker*2-1}; data{speaker*2}];
    N = size(X,1);
    D = size(X,2);
    
    %% Step 1: Initialize pi, mu, sigma        
    clusters = cell(4,K);
    
    % Each of the 64 clusters has a (1x1) pi, (1x20) mu, and (20x20) cov
    for i = 1 : K
        clusters{1,i} = 1/K;
        clusters{2,i} = rand(1,D);
        
        temp = randn(20,20);
        clusters{3,i} = temp'*temp;
    end
    
    % Begin training
    for epoch = 1 : n_epochs
        %% Step 2: Expectation (calculate gamma(z_k))
        % Calculate denominator as sum of all numerators
        denominator = zeros(N,1);
        for k = 1 : K 
            pi_k  = clusters{1,k};
            mu_k  = clusters{2,k};
            cov_k = clusters{3,k};
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Convert all non-positive eigenvalues to force positive definite %
            [V,D2] = eig(cov_k); 
            d = diag(D2);        % get the eigenvalues
            d(d<=1e-7) = 1e-7;   % swap everything below eps with eps
            D_c = diag(d);       % build the "corrected" diagonal matrix "D_c"
            cov_k = (V*D_c*V');  % recalculate new cov_k
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            numerator = pi_k * mvnpdf(X, mu_k, cov_k);

            % Build denominator over iterations (sum of all numerators)
            denominator = denominator + numerator;

            % Save each of N numerators
            clusters{4,k} = numerator;
        end

        % Get gammas (divide numerators by denominator), save them, remove NaNs
        for k = 1 : K  
            clusters{4,k} = clusters{4,k} ./ denominator;
            clusters{4,k}(isnan(clusters{4,k}))=0;
        end

        %% Step 3: Maximization
        for k = 1 : K
            gamma_k = clusters{4,k};
            cov_k = zeros(D);
            N_k = sum(gamma_k);

            % Revised parameters
            pi_k = N_k / N;  
            mu_k = sum(gamma_k .* X) / N_k;

            for n = 1 : N
                diff = ( X(n,:)-mu_k )';
                cov_k = cov_k + (gamma_k(n) .* (diff * diff'));
            end
            cov_k = cov_k / N_k;
            
            % Save out new params
            clusters{1,k} = pi_k;
            clusters{2,k} = mu_k;
            clusters{3,k} = cov_k; 
        end

        %% Step 4: Log Likelihood
        LL = sum(log(denominator)); % second summation already calculated in E-step: 'denominator' variable
        likelihoods(epoch,speaker) = LL; % keep track over epochs and speakers
        
        % Print info
        disp(['Epoch: ', num2str(epoch), ' Likelihood: ', num2str(LL)])
    end
    
    %% Step 5: Save params for current speaker's model into 'speakerModels'
    
    % Create 2D matrix of mu's for speaker's model
    mu_final = zeros(K,D);
    for k = 1 : K
        mu_final(k,:) = clusters{2,k}';
    end
    
    % Create 3D matrix of cov's for speaker's model
    sigma_final = zeros(D,D,K);
    for k = 1 : K
        sigma_final(:,:,k) = clusters{3,k};
    end
    
    % Create vector of mu's for speaker's model
    pi_final = zeros(K,1);
    for k = 1 : K
        pi_final(k) = clusters{1,k};
    end
    
    % Make sure all weights sum to 1
    test = sum(pi_final);
    disp(['Final weights for speaker #', num2str(speaker), ' sum to: ', num2str(test), newline]);
    
    % speakerModels contains the final output parameters
    speakerModels{1,speaker} = pi_final;
    speakerModels{2,speaker} = mu_final;
    speakerModels{3,speaker} = sigma_final;
    
    %% Plot convergance of LL (grader can comment this out to suppress fig output which already in 'results' folder)
    figure(speaker);
    plot(likelihoods(:,speaker))
    title(['Speaker ', num2str(speaker)])
    xlabel('Epochs')
    ylabel('Log Likelihood')
    ylim([likelihoods(2,speaker), likelihoods(n_epochs,speaker)])
    
    %% Save out figure
    % saveas(gcf, sprintf('results/LL_training_plots/logLik_speaker%02.0f.png', speaker))
    
    %% Use fitgmdist() to confirm results during testing
    % GMModel = fitgmdist(X,k);
    
end
% Save the output parameters so don't have to run again
save('speakerModels.mat','speakerModels')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Part 2: Classification
% Compute log-likelihood of test data given the trained speaker parameters

%% Get test data
data2 = cell(10,2);
folder = dir('data/test/');
for files = 1 : length(folder)
    if folder(files).bytes < 1 % if its not real data, skip
        continue
    else
        % fill up 'data' matrix with data and filenames
        data2{files-2, 1} = importdata(folder(files).name);
        data2{files-2, 2} = folder(files).name;
    end
end

% Fix the order of data because speaker 10 was in second row 
[data2{11,1}, data2{11,2}] = data2{2,:};
temp = cell(10,2);
switch1 = 0;
for i = 1 : length(data2)
    if i==2
        switch1 = 1;
    elseif switch1==1
        [temp{i-1,1}, temp{i-1,2}] = data2{i,:};
    else
        [temp{i,1}, temp{i,2}] = data2{i,:};
    end
end
data2 = temp;
clear temp

%% Compute LL of test data given each train speaker params: argmax_j(log(P_x))
load('speakerModels')
K = 20;
% Fill this matrix with loglikelihoods for each train/test speaker pair
LLs = zeros(10,10);
for train_model = 1 : 10 
    for test_speaker = 1 : 10 
        X = data2{test_speaker};
        denominator = zeros(length(X),1);

        for k = 1 : K 
            pi_k  = speakerModels{1,train_model}(k,1);
            mu_k  = speakerModels{2,train_model}(k,:);
            cov_k = speakerModels{3,train_model}(:,:,k);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Convert all non-positive eigenvalues to force positive definite %
            [V,D2] = eig(cov_k); 
            d = diag(D2);        % get the eigenvalues
            d(d<=1e-7) = 1e-7;   % swap everything below eps with eps
            D_c = diag(d);       % build the "corrected" diagonal matrix "D_c"
            cov_k = (V*D_c*V');  % recalculate new cov_k
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            numerator = pi_k * mvnpdf(X, mu_k, cov_k);

            % Build denominator over iterations (sum of all numerators)
            denominator = denominator + numerator;
        end

        % Calculate log-likelihood as sum(ln(sum(numerators))) which is:
        LL = sum(log(denominator)); 
        LLs(test_speaker, train_model) = LL;
    end
end

%% Get max log-likelihood and make prediction
load('data/utt2spk')
trainingNames= ["101188-m", "102147-m","103183-m", "106888-m","110667-m", "2042-f","3424-m", "4177-m","4287-f", "7722-f"];

% IDXS gives the column of max LL for each test set, so first value in IDXS
% implies that 'test1' dataset is from training speaker IDXS(1)
[MAXS,IDXS]= max(LLs,[],2);


%% Calculate posterior probabilities for the GMMs returned by fitgmdist() to compare with results
matlabLL = zeros(10,10);
test_predictions = cell(10,3);
for i = 1 : 10 % test sets
    X_train = [data{i*2-1}; data{i*2}]; 
    GMModel = fitgmdist(X_train,64,'RegularizationValue',0.1);
    for j = 1 : 10
        X_test = data2{j};
        [P,nlogL] = posterior(GMModel, X_test);
        matlabLL(j,i) = -nlogL;
    end
    
    test_predictions(i,1) = data2(i,2);
    test_predictions(i,2) = {trainingNames(IDXS(i))};
end

% Once we have all the columns, take the max of each row.
% That max column is the test-dataset's max.
[~,idx] = max(matlabLL,[],2);
for i = 1 : 10
    test_predictions(i,3) = {trainingNames(idx(i))};
end

% Save tp text file in 'results/' folder
save('results/test_predictions.txt','test_predictions')

%% Accuracy
x = 0;
for i = 1 : 10
    if test_predictions{i,2}==test_predictions{i,3}
        x = x + 10;
    end
end
disp("Classification Accuracy Comparison = " +x+ "% ")


