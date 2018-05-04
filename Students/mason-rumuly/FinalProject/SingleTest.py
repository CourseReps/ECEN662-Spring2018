from DistributionLibrary.DistributionList import DistList
import matplotlib.pyplot as plt
from ObjectiveFunctions import kolmogorov_smirnov, anderson_darling, von_mises

# set up constants
dl = DistList()

# set up example
real = dl[0]()

# generate a set of samples
sample = real.get_sample(1000)

# estimate others
ga = dl[0]().estimate(sample)
gb = dl[1]().estimate(sample)
gc = dl[2]().estimate(sample)
gd = dl[3]().estimate(sample)

# eval objective
kl = kolmogorov_smirnov
print(kl(sample, real), kl(sample, ga), kl(sample, gb), kl(sample, gc), kl(sample, gd))

print(dl[2]().cdf(10))
