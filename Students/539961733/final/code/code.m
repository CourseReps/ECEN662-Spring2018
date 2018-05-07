function array=code()
%input the message and change to binary
prompt = 'What is the string?' ;
message=input(prompt,'s');
n=length(message);
m=n*8;
array=zeros(1,m);
for i=1:n;
   messagenum=abs(message(i));
   messagebin=dec2bin(messagenum,8);
   for j=1:8;
     array((i-1)*8+j)=str2num(messagebin(j));
   end;
end;