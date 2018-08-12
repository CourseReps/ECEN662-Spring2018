%pomdp( p_s,i_dist,pi_start )
p_s=[0.2,0.4,0.2,0.2] % Binomial output (ie:1,0) based on state probability
i_dist=[0.0519,0.3111,0.2222,0.4148]  % Initial distribution of the chain
i_dist=i_dist./sum(i_dist)
%p=[0.2,0.4,0,0.4;0,0,0.5,0.5;0,0,0.3,0.7;0.1,0.7,0,0.2];

p=[0,1,0,0;1,0,0,0;0,0,0,1;0,0,1,0];  % Transition matrix
pi_start=[0.0519,0.3111,0.2222,0.4148]; % POMDP initial state
pi_start=pi_start./sum(pi_start)  

nos=500               % Chain length
pomdp( p_s,i_dist,pi_start,nos,p ) % function call