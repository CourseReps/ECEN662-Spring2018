function [y1,y2]=risecos(x1,x2,fd,fs)
%use raised cosine roll-off function as modulating signal
[yf,tf]=rcosine(fd,fs,'fir/sqrt');
[yo1,to1]=rcosflt(x1,fd,fs,'filter/fs',yf );
[yo2,to2]=rcosflt(x2,fd,fs,'filter/fs',yf );
y1=yo1;
y2=yo2;
