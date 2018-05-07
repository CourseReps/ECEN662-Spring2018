function y=insert_value(x,ratio)
%insert value in digital signal
y=zeros(1,ratio*length(x));
a=1:ratio:length(y);
y(a)=x;

