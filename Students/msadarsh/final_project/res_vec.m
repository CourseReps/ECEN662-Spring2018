function [ res_vec ] = res_vec( prob_vec )
%Used to generate the vector to resolve uniform rand sample
%   
S = 'displaying prob_vec';
% disp(S)
% disp(prob_vec)

no_state=length(prob_vec);

res_vec(1:no_state-1)=0;

val_h=0;

for i=1:1:length(res_vec)
val_h=prob_vec(i)+val_h ;
res_vec(i)=val_h ;
    
    
end 


end

