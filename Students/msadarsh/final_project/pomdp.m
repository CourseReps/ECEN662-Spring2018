function [ avg_norm ] = pomdp( p_s,i_dist,pi_start,nos,p )
%This function simulates the markov chain and also does POMDP estimation
%and calculates the L_2 distance between the estimated and actual
%probability state of the system

p1=0.2;
p2=0.4;
p3=0.6;
p4=0.8;

num_states=4

%p_s=[0.2,0.4,0.6,0.8]; % probability of occurence of 1
%p_s=[1,1,1,1]
p_sc=1.-p_s ; % probability of occurence of zero

D1=diag(p_s);
D0=diag(p_sc);




v=(p^100);

disp('steady state values')
ss=v(1,:)


%%%%%%%%%%%%%%%%%%
%i_dist=ss % change this if needed 
i_dist=i_dist./sum(i_dist)
%pi_start=ss
pi_start=pi_start./sum(pi_start)

des_lim=[0,0,0,1] ;
des_lim(1)=i_dist(1) ;
des_lim(2)=i_dist(2)+des_lim(1) ;
des_lim(3)=i_dist(3)+des_lim(2) ;

des_lim ; % This is for initial starting state
%%%%%%%%%%
%nos=100;  % no of steps of markov chain 

ms_a=10000; % markov state accuracy. - No of times the chain has to be run to
          % get decend probability distribution

m_sc(1:nos,1:4)=0 ;% Markov state count

for j=1:1:ms_a
    
init_state =  init_state_val( i_dist ) ;
crt_state = init_state ;
    

for i=1:1:nos

nxt_state=nxt_state_eval(crt_state,p)  ;         

m_sc(i,nxt_state)=m_sc(i,nxt_state)+1;

crt_state=nxt_state ;

% determining the markov state estimation 


end 

end 


row_sum=sum(m_sc,2);

row_sum_i=1./row_sum;

row_sum_mat=diag(row_sum_i);

m_state_tprob=row_sum_mat*m_sc ;


% Pomdp noise estimation 

% pi_start=ss

% pi_start=i_dist+0.2.*[1,1,1,1] ;
% pi_start=pi_start./sum(pi_start)

init_state =  init_state_val( pi_start ) ;
crt_state = init_state ;
nxt_state=nxt_state_eval(crt_state,p)  ;  
crt_state=nxt_state ;
pi_prev= pi_start;

estimate_array(1:nos,1:num_states)=0;

for i=1:1:nos

% sampling 
observation= rand_bin_eval(p_s(crt_state));
if observation==1
    pi_est=pi_prev*p*D1 ;
else
    pi_est=pi_prev*p*D0;
end 

pi_est_sum=sum(pi_est);
pi_est=pi_est./(pi_est_sum) ;% normalizing for probability

pi_est=pi_est;

for j=1:1:num_states
estimate_array(i,j)=pi_est(j);

end 

pi_prev=pi_est;

nxt_state=nxt_state_eval(crt_state,p)  ;         
crt_state=nxt_state ;    
    
    
    
% markov simulation
end 


estimate_array;

% disp('m_sc printing')
% disp(m_sc(1,:))
% disp(m_sc(2,:))

% norm calculation

norm=[];
for k=1:1:nos
% estimate_array,  m_state_tprob   
val=0;
for m=1:1:num_states
%val=val+((estimate_array(k,m)*m_state_tprob(k,m)));
val=val+estimate_func(estimate_array(k,m),m_state_tprob(k,m));

end
%val= sqrt(val);
norm=[norm,val]  ;  
end 

norm;
plot([1:nos],norm);
disp('avg_norm val')
avg_norm= sum(norm)/nos 





end

