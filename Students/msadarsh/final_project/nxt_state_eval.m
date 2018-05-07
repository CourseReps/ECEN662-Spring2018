function [ nxt_state ] = nxt_state_eval(crt_state,p_mat ) % p_mat is probability matrix
%Next state based on transition probability matrix

p_res=res_vec(p_mat(crt_state,:));

uni_val=rand;

% pv= 'printing nxt state init'
p_mat_crt=p_mat(crt_state,:);
nxt_state= length(p_mat_crt);

for i=1:1:length(p_res)
if(uni_val<=p_res(i))
nxt_state=i;    
break
end 
end 

end

