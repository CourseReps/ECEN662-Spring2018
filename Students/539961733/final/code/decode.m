function message=decode(bin)
%translate binary to message
n=length(bin);
m=n/8;
for i=1:m;
messagestr=num2str(bin((i-1)*8+1:(i-1)*8+8));
messagebin=bin2dec(messagestr);
messagechar=char(messagebin);
message(i)=messagechar;
end;