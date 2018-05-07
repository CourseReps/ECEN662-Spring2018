function [ res_val ] = rand_bin_eval( p )
% Here p is the probability of 1
res_val=0;
if rand<=p
    res_val=1;
else
    res_val=0;
end 

end

