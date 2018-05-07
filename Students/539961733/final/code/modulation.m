function [y1,y2]=modulation(x)
%16QAM get x-y source 
N=length(x);
a=1:2:N;
y1=x(a);
y2=x(a+1);                 
a=1:2:N/2; 
temp1=y1(a);
temp2=y1(a+1);
y11=temp1*2+temp2;
temp1=y2(a);
temp2=y2(a+1);
y22=temp1*2+temp2;          %2 to 4 
y1=(y11*2-1-4)*1.*cos(2*pi*a);
y2=(y22*2-1-4)*1.*cos(2*pi*a);%modulate phase
y1(find(y11==0))=-3;
y1(find(y11==1))=-1;
y1(find(y11==3))=1;
y1(find(y11==2))=3;
y2(find(y22==0))=-3;
y2(find(y22==1))=-1;
y2(find(y22==3))=1;
y2(find(y22==2))=3;

