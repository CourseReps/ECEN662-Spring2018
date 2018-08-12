function [ nxt_state ] = init_state_val( init_dist )
%Gives the initial state of the chain
% can be used for any random resolution


p_res=res_vec(init_dist);

uni_val=rand;

% uni_val=rand;

% pv= 'printing nxt state init'

nxt_state= length(init_dist);

for i=1:1:length(p_res)
if(uni_val<=p_res(i))
nxt_state=i;    
break
end 
end 



end

