The code uses the fact that MMSE is minimized when the estimate is E[theta|Y]

If Y is a 1D Beta-Binomial output, then E[theta|Y]= (Y+2)/47

Given a vector Y it is very difficult to get E[Theta|Y vector]

The binomial distribution represents a Gaussian in the limiting case. Hence this helps in visualization of the distribution 
with mean np and how it progresses as p increases

Hence an particular way of estimation is to use Mean of the samples of the output and to estimate as E[mu|Theta] so as to remove the 
variations in the output
