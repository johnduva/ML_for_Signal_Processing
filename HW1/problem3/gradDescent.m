function [final_weight01, final_weight0001] = gradDescent(w0,Edif,n)
% Inputs:
% w0 = initial weight matrix (should be all ones in this case)
% eta = learning rate
% n = number of iterations

% Output:
% final_weight = weight matrix at the end of 'n' iterations
% Ensure the max(W,0) constraint used in each iteration

    final_weight = {w0,w0,w0,w0};
    count = 1;

    for eta = [0.0001, 0.001, 0.01, 0.05]
        for i = 1 : n
            final_weight(count) = max(final_weight{count} - eta*Edif, 0);
        end
        count = count + 1;
    end
   
    
    
end