function [ ef_rv ] = estimate_func( a,b )
%estimate_array(k,m),m_state_tprob(k,m)
% This is the function to estimate the error. Used here is the L_2 distance
ef_rv=(a-b)^2;
%ef_rv=(1-(a/b))^2;
end

