import numpy as np

def fit_gaussian(data):
    exp = np.ndarray(shape=(2))
    exp[0] = np.mean(data[:, 0])
    exp[1] = np.mean(data[:, 1])

    y = data
    covariance = np.cov(np.transpose(data))

    print("mean = [%s, %s]" %(exp[0], exp[1]))
    print("Covariance =")
    print(covariance)

    # stddev = [np.sqrt(covariance[0, 0]), np.sqrt(covariance[1, 1])]

    return exp, covariance


def evaluate_2d_gaussian(mean, cov, data, A=1):
    # using the evaluation of the elliptical 2d gaussian from here:
    # https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    a = cov[0, 0]
    b = cov[0, 1]
    c = cov[1, 1]
    x0 = mean[0]
    y0 = mean[1]

    dataRange = data.shape[0]
    answers = np.ndarray(shape=(dataRange))

    for d in range(0, dataRange):
        x = data[d, 0]
        y = data[d, 1]
        # TODO: note this could be optimized easily
        result = A * np.exp(-(a*pow((x - x0), 2) + 2*b*(x - x0)*(y - y0) + c*pow((y - y0), 2)))

        answers[d] = result

    return answers