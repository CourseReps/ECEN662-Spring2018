clear;
snr=5;
%%%%%%%%%%%%%%change the message to binary%%%%%%%%%%%%%%%%%%%%%%%
message=code();
message=[message 0 0 ]; % confirm the last 2 bits are 0 
%%%%%%%%%%%%%%code message with convolutional code%%%%%%%%%%%%%%%
source=conencode(message,length(message));
%%%%%%%%%%%%%16QAM to simmulate the channel%%%%%%%%%%%%%%%%%%%%%%
[source1,source2]=modulation(source);
%insert value
sig_insert1=insert_value(source1,8);
sig_insert2=insert_value(source2,8);
%modulating signals
[source1,source2]=risecos(sig_insert1,sig_insert2,0.25,2);
%add noise
[x1,x2]=noise(source1',source2',snr);
%%%%%%%%%%%%%demodulate signals%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig_noise1=x1';
sig_noise2=x2';
%multiply with modulating signal
[sig_noise1,sig_noise2]=risecos(sig_noise1,sig_noise2,0.25,2);
%pick the peaks of signals
[x1,x2]=pick(sig_noise1,sig_noise2,8);
sig_noise1=x1;
sig_noise2=x2;
%demodulate signals
signal=demodulate(sig_noise1,sig_noise2);
%%%%%%%%%%%%%decode the signal with Viterbi method%%%%%%%%%%%
message1=viterbi(signal,length(message));
errorrate=length(find((source-signal)~=0))/length(source)
viterbierror=length(find((message1-message)~=0))/length(message)
%%%%%%%%%%%%%translate binary to message%%%%%%%%%%%%%%%%%%%%%
decode(message1(1:length(message)-2))