function [y1,y2]=pick(x1,x2,t)
%pick the peak of signals
y1=x1(t*3*2+1:t:(length(x1)-t*3*2));
y2=x2(t*3*2+1:t:(length(x1)-t*3*2));