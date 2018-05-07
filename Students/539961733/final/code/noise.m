function [y1,y2]=noise(x1,x2,snr)
%add noise in signal
snr1=snr+10*log10(4);   %symbol SNR
ss=var(x1+i*x2,1);
y=awgn([x1+j*x2],snr1+10*log10(ss/10),'measured');
y1=real(y);
y2=imag(y);