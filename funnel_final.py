import numpy as np
import matplotlib.pyplot as plt
import math

def FSim1(n, lamb):
    """
    a function to simulate a funnel

    :param n: number of users to simulate
    :param lambda: the paramter for the random exponential distribution

    :return list: exponential random simulations
    """
    return np.random.exponential(scale=1.0/ lamb, size=n)


def FSim2(user_times, breaks):
    """
    groups the users' times into bins and returns a list of counts for each bin

    :param user_times: list of user quit times
    :param breaks: list of intervals breaks

    :return histogram: list of frequencies for each bin
    """
    bins_edges = [0] + breaks + [np.inf]
    return list(np.histogram(user_times, bins=bins_edges)[0])


def EstLam1(user_times):
    """
    compute lambda given an array of user quit times
    lambda = 1/mean(userstimes)

    :param user_times: array of user quit times
    :return lambda: float
    """
    return 1.0 / np.mean(user_times)



def plot_funnel(n, lamb, start, end, inc, showPlot=True, title=""):
    """
    a function to plot a funnel based on simulated users from a
    random expoential distribution

    :param n: number of users to simulate
    :param lamb: lambda parameter for the random exponential number generator
    :param start: the beginning of the stops
    :param end: the last stop
    :param inc: the interval size between stops

    :return plot:
    """
    # -- Simulation
    # simulate n users using FSim1 function and create the stops with the end
    # at `end`+`inc` so that stops array includes `end` as the last stop
    randos = FSim1(n, lamb)
    stops = np.arange(start, end+inc, inc)

    # -- Main Loop
    # for each stop, count how many users are above or at the stop threshold and
    # store the count. A list comprehension is used because its faster than
    # numpy summations of vectors when repeated many times.
    funnel_counts = [sum([1 for r in randos if r >= stop]) for stop in stops]

    # -- Plotting
    # plot a bar chart with the given funnel_counts as heights,
    # the stops as the left edges and a width of .1 so that you can see
    # whitepace between bars. Also adds what lambda value is used to generate
    # the random user and place itthe coordinates (x=last stop, y=90% of n users)
    plt.bar(stops, funnel_counts, width=.1, label='$\lambda$ = %s' % lamb)
    plt.xlabel("Survival Time")
    plt.ylabel("Number of Users Who Survived")
    plt.title(title)
    if showPlot:
        plt.show()
    return


def compute_CI(user_times, n_boots, a):
    """
    compute the confidence interval for the lambda parameter
    the governs the distribution of user_times using bootstraping

    :param user_times: list or array of user_times
    :param n_boots: int for number of bootstraps to use
    :param a: float for alpha level to use for the CI

    :return confi interval: tuple of lower and upper bounds as floats
    """
    # -- Main Loop bootstrap
    # in each iteration, compute the lambda estimate and
    # store it so that we can extract the a/2 and 1-a/2 bounds
    lam_estimates = [EstLam1(np.random.choice(user_times, len(user_times), replace=True) )
                     for i in range(n_boots)]
    return tuple(np.percentile(lam_estimates, [a/2.0 * 100, (1.0-(a/2)) * 100]))


# Q 3(d)
def PlotLambduhCI(numlist):
    """
    @param:
    numlist: a list of number of users
    """
    lambduhs = []
    lbounds = []
    ubounds = []
    for num in numlist:
        quit_time = FSim1(num, 1)
        lambduhs.append(EstLam1(quit_time))
        lbounds.append(compute_CI(quit_time, 500, 0.05)[0])
        ubounds.append(compute_CI(quit_time, 500, 0.05)[1])
    plt.plot(numlist, lbounds, linestyle=':', marker='o', label='Lower Bound')
    plt.plot(numlist, ubounds, linestyle=':', marker='o', label='Upper Bound')
    plt.plot(numlist, lambduhs, linestyle=':', marker='o', label='$\hat{\lambda}$')
    plt.xlabel('Number of Users')
    plt.ylabel('Survial Time')
    plt.legend(loc='best')
    plt.title('Q3: 95% CI As Sample Size Increases')
    plt.show()


def F(x, lamb):
    """
    CDF for expoential distribution
    """
    return 1 - math.exp(-lamb * x)


def compute_loglik(histogram, bps, lambduh):
    """
    compute the loglikelihood for a given lambduh

    this verion of likelihood is used when sensor data used, and breakpoints are
    used as quit-times instead of users actual quit times
    :param histogram: array or list of int representing the number of users surviving
                        each breakpoint. The first value in the array represents
                        the number of users who did not make it past the first breakpoint
    :param bps: array of floats or ints representing stopping points for each event
    :param return: loglikelihood - float
    """
    # There are three parts to compute this log likelihood
    # 1. users who did not survive past the first event
    # 2. users who survived all events
    # 3. users who survived any amount excluding users who survied all
    # what is returned is the sum of loglikelihoods for each of the parts listed above,
    # but 3. is iterative, so it is placed in its own variable.
    loglik_any = sum([np.log(F(bps[i+1], lambduh)-F(bps[i], lambduh)) for i in range(0, len(bps)-1)])
    return ( histogram[0] * np.log(F(breaks[0], lambduh)) +
        histogram[-1] * -lambduh * breaks[-1] + loglik_any )


def MLE1(histogram, breaks):
    """
    create a lambda function that is able to compute the loglikelihood of a given
    lambda and histograms

    :param histogram: array or list of int representing the number of users surviving
                        each breakpoint. The first value in the array represents
                        the number of users who did not make it past the first breakpoint
    :param breaks: array of floats or ints representing stopping points for each event
    :return lambda function: this lambda function accepts a lambda parameter estimate
                        and returns the loglikelihood
    """
    return lambda x: compute_loglik(histogram, breaks, x)


def MaxMLE(histogram, breaks, lambda_range):
    """
    function to find the lambda that maximizes the loglikelihood

    :param histogram: array or list of int representing the number of users surviving
                    each breakpoint. The first value in the array represents
                    the number of users who did not make it past the first breakpoint
    :param breaks: array of floats or ints representing stopping points for each event
    """
    # - Dictionary comprehension
    # hash each lambda and its corresponding loglikelihood to find the max
    # and easily extract it from dictionary using the max loglikelihood as key
    logliks = {MLE1(histogram, breaks)(l):l for l in lambda_range}
    
    #logliks={prob1:lambda1, prob2:lambda2}
    return logliks[max(logliks.keys())]


if __name__=="__main__":

    # - Q1.
    plot_funnel(1000, 2, .25, 3, .25, title="Q1.A: Funnel Visualization $\lambda$ = 2")

    lamb_to_try = np.arange(.2, 3.2, .2)
    for  l in lamb_to_try:
       plot_funnel(1000, l, .25, 3, .25, showPlot=False, title="Q1.B: Funnel Visualization Over $\lambda$ = [.2, 3]")
    plt.show()

    # - Q2
    print "----- Question 2 --------"
    print FSim2([0.2, 0.4],[0.25, 0.5])
    print "\n"

    # - Q3
    print "----- Question 3 --------"
    quit_time = FSim1(1000, 1)
    print "95% CI for the estimated lamb is:", compute_CI(quit_time, 500, .05)
    print "Estimated lamb is:", EstLam1(quit_time)
    print "\n"

    PlotLambduhCI([100, 200, 500, 1000, 2000, 5000, 10000])

    # - Q4
    print "----- Question 4 --------"
    x = [0.25, 0.45, 0.75]
    breaks = [0.5]
    PRT = MLE1(FSim2(np.array(x), breaks), breaks)
    print "4.C:"
    print PRT(1)
    print "\n"

    print "4.D:"
    print MaxMLE(FSim2(np.array(x), breaks), breaks, np.arange(0.1, 3, 0.05))
    print "\n"

    # - Q5.
    print "----- Question 5 --------"
    breaks_list = [[.25, .75], [.25, 3], [.25, 10]]

    for breaks in breaks_list:
        diff = []
        for i in range(1000):
            users = FSim1(100, 1)
            diff.append(EstLam1(users) - MaxMLE(FSim2(users, breaks), breaks, np.arange(.1, 3, .05)))
        print "breaks:", breaks, "\tmean difference:", np.mean(diff)
