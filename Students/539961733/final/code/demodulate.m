function y=demodulate(x1,x2)
%demodulate signals
xx1(find(x1>=2))=3;
xx1(find((x1<2)&(x1>=0)))=1;
xx1(find((x1<0)&(x1>=-2)))=-1;
xx1(find(x1<-2))=-3;
xx2(find(x2>=2))=3;
xx2(find((x2<2)&(x2>=0)))=1;
xx2(find((x2<0)&(x2>=-2)))=-1;
xx2(find(x2<-2))=-3;
%10 to 2£¬eg:xx1=-3,temp1=00
%xx1=-1,temp1=01;xx1=1,temp1=11;xx1=3,temp1=10
temp1=zeros(1,length(xx1)*2);
temp1(find(xx1==-1)*2)=1;
temp1(find(xx1==1)*2-1)=1;
temp1(find(xx1==1)*2)=1;
temp1(find(xx1==3)*2-1)=1;
%the second signal
temp2=zeros(1,length(xx1)*2);
temp2(find(xx2==-1)*2)=1;
temp2(find(xx2==1)*2-1)=1;
temp2(find(xx2==1)*2)=1;
temp2(find(xx2==3)*2-1)=1;
%converge 2 signals to 1 signal
y=zeros(1,length(temp1)*2);
y(1:2:length(y))=temp1;
y(2:2:length(y))=temp2;
