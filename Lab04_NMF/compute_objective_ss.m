function [obj] = compute_objective_ss(V, W, B, alpha, beta) 
    V = V + eps;
    obj = sum(sum(V.*log(V./(B*W)))) + sum(V, 'all') - sum(sum(B*W)) + alpha*norm(W,1) + beta*norm(B,1);
end
