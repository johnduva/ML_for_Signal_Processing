%%  K-Means function to generate K clusters of an input image
% Output:
%  - final set of coordinates of the K centroids
%  - final segmented image based on the K centroids

function [C, segmented_image] = KMeans(Image,K,maxIter)
%% 1. Image vectorization based on RGB components
X = Image;
dim1 = size(X,1); dim2 = size(X,2); flat = dim1*dim2;
X = reshape(X, [flat, 1 3]);
%% 2. Intial RGB Centroid Calculation
% Sohaib said this was the same as step 3 so I'm conjoining them
%% 3. Randomly initializing K centroids (select those centroids from the actual points)
centroids = ones(3,K);
% for each centroid (column)
for i = 1:K
    % for each dimension
    for dim = 1 : 3
        % initialize K number of centroids (with a value between 0 and max)
        num = rand(1) * max(X(:,:,dim));
        % make sure this centroid does not already exist
        while ismember(num, centroids)
            num = rand(1) * max(X(:));
        end

        centroids(dim,i) = num;
    end
end

%% 4. Assign data points to K clusters using the following distance - dist = norm(C-X,1)
for j = 1 : maxIter
    % get the distances from each 3D pixel to each of the K centroids: dist=(154401 x K)
    dist = ones(length(X), K);
    for cents = 1:K
        for pixel = 1 : length(X)
            for dim = 1 : 3
                % get the euclidean distance of that pixel 
                dist(pixel, cents) = sqrt(sum((centroids(:,cents) - double(squeeze(X(pixel, 1, :)))).^2));
            end
        end
    end

    % determine which centroid is closer & assign each pixel a group from 1:K 
    cluster = ones(length(X),1);
    for distance = 1 : length(dist)
        [MIN IDX] = min(dist(distance,1:K));
        % each row in 'cluster' tells you which cluster that pixel is in
        cluster(distance) = IDX;
    end

    %% 5. Re-computing K centroids
    newDist = centroids;
    for x = 1: 3
        for y = 1 : K
            newDist(x,y) = mean(  X(cluster==y,1,x)  );
        end
    end
    % determine which (if any) centroids didn't have any associated data (NaNs)
    nans = find(isnan(newDist(1,:)));

    % use similar for-loop to replace NaNs in newDist
    for i = nans
        for dim = 1 : 3
            num = rand(1) * max(X(:,:,dim));
            while ismember(num, centroids); num = rand(1) * max(X(:)); end
            newDist(dim,i) = num;
        end
    end
    % update 'centroids' to reflect new distances as you iterate
    centroids = newDist;

% Reiterate through steps 4 and 5 until maxIter reached. Set maxIter = 100
end

clusterFinal = ones(length(cluster),1,3);
for pix = 1 : length(cluster)
    for dim = 1 : 3
        clusterFinal(pix,1,dim) = centroids(dim, cluster(pix));
    end
end

% Return K centroid coordinates and segmented Image
C = centroids;
segmented_image = uint8(reshape(clusterFinal, [dim1 dim2 3]));
end
