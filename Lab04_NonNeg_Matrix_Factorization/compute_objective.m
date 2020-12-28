function [obj] = compute_objective(V, B, W) 
    % KL Divergence
    V = V + eps;
    obj = sum(sum(V.*log(V./(B*W)))) + sum(V, 'all') - sum(sum(B*W)); 
end