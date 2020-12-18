function [B,W,obj,k] = nmf(V,rank,max_iter,lambda)
    % Initialize B and W randomly
    [D, N] = size(V);
    B = rand(D,rank);
    W = rand(rank,N);
    % make sure W has unit-sum columns (each column should sum to 1)
    W = W./sum(W);
    
    % Calculate the initial objective (define a new function for sparse NMF)
    now = [];
    next = [];
    for k = 1 : max_iter
        % if we've already established a second objective...
        if ~isempty(next)
            % use that second objective as the first
            now = next;
        else % if this is our first time running through, calc initial obj
            now = compute_objective(V, B, W);
        end
        
        % update B and W
        B = B .* ((V./(B*W))*W' ./ (ones(size(V))*W') );
        W = W .* ((B'*(V./(B*W))) ./ (B'*ones(size(V))) );
        % 'next' is the new objective
        next = compute_objective(V, B, W);
        % check to see if you've met your thresh yet
        if abs(next-now) < lambda; break; end
        % output 'next'
        obj = next;
    end

end
