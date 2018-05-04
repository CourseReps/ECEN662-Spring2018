from DistributionLibrary.DistributionList import DistList
import matplotlib.pyplot as plt
from ObjectiveFunctions import kolmogorov_smirnov, kuiper, anderson_darling, von_mises

# set up constants
dl = DistList()
sample_sizes = [8, 10, 20, 40, 80, 100, 200, 400, 800, 1000]
trials = 100  # number of sample sets to try per size
objectives = [kolmogorov_smirnov, kuiper, anderson_darling, von_mises]
ons = ['Kolmogorov-Smirnov Test', "Kuiper's Test", 'Anderson-Darling Test', 'Cramér–von Mises Criterion']

# set up receptacles
# Index meaning: objective, origin, guess, sample size
results = [[[[0 for i in range(len(sample_sizes))
              ] for j in range(len(dl))
             ] for k in range(len(dl))
            ] for l in range(len(objectives))
           ]
# Index meaning: origin, objective, sample size
accuracy = [[[0 for i in range(len(sample_sizes))
              ] for j in range(len(objectives))
             ] for k in range(len(dl))]

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

                    best = [[-1, -1] for o in range(len(objectives))]

                    for g in range(len(dl)):

                        # estimate distribution
                        guess = dl[g]().estimate(sample[:sample_sizes[s]])

                        # save goodness of fit; averaged across all trials
                        for o in range(len(objectives)):
                            stat = objectives[o](sample[:sample_sizes[s]], guess)/trials
                            results[o][r][g][s] += stat

                            if (best[o][0] < 0) or (stat < best[o][0]):
                                best[o][0] = stat
                                best[o][1] = g

                    # eval trial for this size
                    for o in range(len(objectives)):
                        if best[o][1] == r:
                            accuracy[r][o][s] += 1/trials

                success = True

            except ValueError:
                print('Retry Trial ' + str(t))

    # plot values
    for o in range(len(objectives)):
        plt.figure()
        plt.title(dl[r].name() + ': ' + ons[o])
        for g in range(len(dl)):
            plt.plot(sample_sizes, results[o][r][g], label=dl[g].name())
        plt.legend(loc='best', shadow=False)
        plt.ylabel('Statistic')
        plt.xlabel('Observations per Sample')
        plt.xscale('log')
        plt.savefig((dl[r].name() + '$' + ons[o] + '.png'), format='png')

    # plot accuracy
    plt.figure()
    plt.title(dl[r].name() + ' Objective Comparison')
    for o in range(len(objectives)):
        plt.plot(sample_sizes, accuracy[r][o], label=ons[o])
    plt.legend(loc='best', shadow=False)
    plt.ylabel('Accuracy')
    plt.xlabel('Observations per Sample')
    plt.xscale('log')
    plt.savefig(dl[r].name() + ' Objective Comparison.png', format='png')
