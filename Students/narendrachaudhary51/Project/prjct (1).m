clear all;
close all

%% gaussian case
%k=[1 2 4 6 8 10 25 50 75 80 85 90 95 100];
k=1:100;
N=100000;
meandiff=3
for j=1:length(k)
for i=1:N
nois1(i)= sum(randn(k(j),1))/sqrt(k(j));
nois2(i)= sum(randn(k(j),1))/sqrt(k(j));  % variance normalised and mean zero
sym1(j,i)=  meandiff+nois1(i);
sym2(j,i)= -meandiff+nois2(i);
a1(j,i)=sym1(j,i)>0;
a2(j,i)=sym2(j,i)<0 | sym2(j,i)<0;
end
det1(j) = (length(find(a1(j,:)==1))+length(find(a2(j,:)==1)))/(2*N);
end
%% detection error 

gauss_error = qfunc(meandiff);

histogram(sym1(2,:),1000,'EdgeColor','b')
hold on
histogram(sym2(10,:),1000,'Edgecolor','k')

figure
grid on
plot(k,1-det1,'r','LineWidth',2)
hold on
grid on
plot(k,gauss_error*ones(1,length(k)),'LineWidth',2)
clear all
%% uniform distribution
a_u=-sqrt(12)/2; b_u=sqrt(12)/2; 

%r = a + (b-a).*rand(100,1);
%k=[1 2 4 6 8 10 25 50 75 80 85 90 95 100];
k=1:50
N=1000000;
meandiff=1
for j=1:length(k)
for i=1:N
nois1(i)= (sum(rand(k(j),1))-.5*k(j))/sqrt(k(j));
nois2(i)= (sum(rand(k(j),1))-.5*k(j))/sqrt(k(j)); % variance normalised and mean zero

sym1(j,i)=  meandiff+nois1(i);
sym2(j,i)= -meandiff+nois2(i);
a1u(j,i)=sym1(j,i)>0;
a2u(j,i)=sym2(j,i)<0 | sym2(j,i)<0;
end
det_n(j) = (length(find(a1u(j,:)==1))+length(find(a2u(j,:)==1)))/(2*N);
end


%% detection error 

gauss_error = qfunc(meandiff*(sqrt(12)));
figure
histogram(sym1(2,:),1000,'EdgeColor','r')
hold on
histogram(sym2(10,:),1000,'Edgecolor','k')
figure
plot(k,1-det_n,'r','LineWidth',2)
hold on
plot(k,gauss_error*ones(1,length(k)),'LineWidth',2)
%% Probability distribution of sum of iid uniform random variables

% irwin hall distribution
k=1:25;
x=linspace(0,16,10000);
f=zeros(length(k),length(x));
err=zeros(1,length(k));
for i=1:length(k)
   for j=1:length(x)
       for k1=0:k(i)
           f(i,j)=f(i,j)+((-1).^k1)*((x(j)-k1).^(k(i)-1))*sign(x(j)-k1)*nchoosek(k(i),k1);
       end
       f(i,j)=(1/(2*factorial(k(i)-1)))*f(i,j);
   end
end
for i=1:10
    grid on
    plot(x,f(i,:),'LineWidth',1.5)
    hold on
end

%% exponential case:

%r = a + (b-a).*rand(100,1);
%k=[1 2 4 6 8 10 25 50 75 80 85 90 95 100];
k=1:100
N=10000;
meandiff=1;
for j=1:length(k)
for i=1:N
    R = exprnd(1,[1,k(j)]);
nois1(i)= (sum(R)-k(j))/sqrt(k(j));
nois2(i)= (sum(R)-k(j))/sqrt(k(j)); % variance normalised and mean zero

sym1(j,i)=  meandiff+nois1(i);
sym2(j,i)= -meandiff+nois2(i);
a1u(j,i)=sym1(j,i)>0;
a2u(j,i)=sym2(j,i)<0 | sym2(j,i)<0;
end
det_n(j) = (length(find(a1u(j,:)==1))+length(find(a2u(j,:)==1)))/(2*N);
end


%% detection error 

gauss_error = qfunc(meandiff);
figure
histogram(sym1(2,:),1000,'EdgeColor','r')
hold on
histogram(sym2(10,:),1000,'Edgecolor','k')
figure
plot(k,1-det_n,'r','LineWidth',2)
hold on
plot(k,gauss_error*ones(1,length(k)),'LineWidth',2)

%% Pdf of iid exponential variable

% erlang distribution

k=1:25;
x=linspace(0,50,10000);
f=zeros(length(k),length(x));
err=zeros(1,length(k));
for i=1:length(k)
   for j=1:length(x)
      f(i,j)=(x(j)^(k(i)-1)*exp(-x(j)))/factorial(k(i)-1);
   end
end
for i=15:25
    grid on
    plot(x,f(i,:),'LineWidth',2)
    hold on
end

%% correlated case: %% gaussian case
%k=[1 2 4 6 8 10 25 50 75 80 85 90 95 100];
clear all
k=1:100;
N=100000;
meandiff=5
for j=1:length(k)
for i=1:N
zz=randn(k(j),1);   h=[.1,1];
y = conv(h,zz);
nois1(i)= sum(y)/sqrt(k(j));
nois2(i)= sum(y)/sqrt(k(j));  % variance normalised and mean zero
sym1(j,i)=  meandiff+nois1(i);
sym2(j,i)= -meandiff+nois2(i);
a1(j,i)=sym1(j,i)>0;
a2(j,i)=sym2(j,i)<0 | sym2(j,i)<0;
end
det1(j) = (length(find(a1(j,:)==1))+length(find(a2(j,:)==1)))/(2*N);
end
%% detection error 

gauss_error = qfunc(meandiff);

histogram(sym1(2,:),1000,'EdgeColor','b')
hold on
histogram(sym2(10,:),1000,'Edgecolor','k')

figure
grid on
plot(k,1-det1,'r','LineWidth',2)
hold on
grid on
plot(k,gauss_error*ones(1,length(k)),'LineWidth',2)
