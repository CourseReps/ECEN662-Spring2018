function [ partition ] = rand_eval( val,val_array )
%rand_eval - Sampling for a paricular probability
partition =length(val_array)+1
for i=1:1:length(val_array)
 if val<= val_array(i)
     partition=i 
     break
 end 
    
 
 
    
end 


end

