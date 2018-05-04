from DistributionLibrary.DistributionList import DistList
import matplotlib.pyplot as plt
from ObjectiveFunctions import kolmogorov_smirnov, kuiper, anderson_darling, von_mises

# set up constants
dl = DistList()
sample_sizes = [8, 10, 20, 40, 80, 100, 200, 400, 800, 1000]
trials = 100  # number of sample sets to try per size
objectives = [kolmogorov_smirnov, kuiper, anderson_darling, von_mises]
ons = ['Kolmogorov-Smirnov Test', "Kuiper's Test", 'Anderson-Darling Test', 'Cramér–von Mises Criterion']

# set up receptacle
# Index meaning: origin, guess, sample
# results = [[[[0] * len(sample_sizes)] * len(dl)] * len(dl)] * len(objectives)
results = [[[[0 for i in range(len(sample_sizes))
              ] for j in range(len(dl))
             ] for k in range(len(dl))
            ] for l in range(len(objectives))
           ]

# do empirical trials
for r in range(len(dl)):

    # generate data for each trial
    for t in range(trials):

        success = False
        while not success:

            try:
                # generate test data
                sample = dl[r]().get_sample(max(sample_sizes))

                # test each distribution with each sample length
                for s in range(len(sample_sizes)):
                    for g in range(len(dl)):

                        # estimate distribution
                        guess = dl[g]().estimate(sample[:sample_sizes[s]])

                        # save goodness of fit; averaged across all trials
                        for o in range(len(objectives)):
                            results[o][r][g][s] = objectives[o](sample[:sample_sizes[s]], guess)/trials + results[o][r][g][s]
                success = True

            except ValueError:
                print('Retry Trial ' + str(t))

    # plot
    for o in range(len(objectives)):
        fig = plt.figure()
        plt.title(dl[r].name() + ': ' + ons[o])
        for g in range(len(dl)):
            plt.plot(sample_sizes, results[o][r][g], label=dl[g].name())
        plt.legend(loc='best', shadow=False)
        plt.ylabel('Statistic')
        plt.xlabel('Observations per Sample')
        plt.xscale('log')
        plt.savefig((dl[r].name() + '$' + ons[o] + '.png'), format='png')
