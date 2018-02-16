import numpy as np

def fit_gaussian(data):
    numDims = data.shape[1]

    print("Fitting mean and covariance for %s dimensions" %(numDims))

    means = []
    for i in range(numDims):
        means.append(np.mean(data[:, i]))

    # cov is symmetric so transpose stuff shouldn't matter here.
    covariance = np.cov(np.transpose(data))

    print("Mean =")
    print(*means)

    print("Covariance =")
    print(covariance)

    return means, covariance


def evaluate_nd_gaussian(mean, cov, data, A=1):
    dims = data.shape[1]
    dataRange = data.shape[0]
    answers = np.ndarray(shape=(dataRange))

    covMat = np.matrix(cov)
    covInv = np.linalg.inv(covMat)
    covDet = np.linalg.det(covMat)

    for d in range(0, dataRange):
        y = np.matrix(data[d, :])
        diff = y - mean
        numerator = -0.5 * diff * covInv * diff.transpose()
        denominator = np.sqrt(pow(2*3.14159265359, dims) * covDet)
        a = A*np.exp(numerator) / denominator
        answers[d] = a

    return answers
