function [B,W,obj,k] = ssnmf(V,rank,max_iter,lambda,alpha,beta) 
    [D, N] = size(V);
    B = rand(D,rank);
    W = rand(rank,N);
    W = W./sum(W);
    
    now = [];
    next = [];
    for k = 1 : max_iter
        if ~isempty(next)
            now = next;
        else 
            now = compute_objective_ss(V, W, B, alpha, beta);
        end
%         B = B .*   ((V./(B*W))*W' ./ (sum(W',1)+beta) );
%         W = W .* ((B'*(V./(B*W))) ./ (sum(B',2)+alpha) );
        B = B .* (  (V./(B*W))*W' ./ (ones(size(V))*W')+beta  ) ;
        W = W .* ((B'*(V./(B*W))) ./ (B'*ones(size(V)))+alpha ) ;
        next = compute_objective_ss(V, W, B, alpha, beta);
        if abs(next-now) < lambda; break; end
        obj = next;
    end
end