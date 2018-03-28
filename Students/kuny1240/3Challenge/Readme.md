## Challenge3
In this problem, since we already know that the distribution is a binomial~(40,theta), so we have a intuition to find an analytical MVU 
estimator in this problem. We already know that Y/N is a unbiased estimator for a binomial distribution, we now prove that __Sum(Y)/K = T(Y)
is a complete statistic__. E(Y) is also a binomial~(40,theta), so: 
1. E_theta(g(T)) = Sigma(n t=0) g(t)C(n,t)p^(t)(1-p)^(n-t) 
2. For 0<theta<1 so that C(n,t),(p/(1-p))^(t) and (1-p)^n can not be 0. 
3. From 2, we get that P(g(t) = 0) = 1, so T(Y) is a complete statistic   
__So that the final estimator can be written in the form T(Y)/N__
