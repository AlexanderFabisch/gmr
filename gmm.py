import numpy as np
from sklearn.utils import check_random_state
from multivariate_normal import MultivariateNormal


class GMM(object):
    def __init__(self, n_components, priors=None, means=None, covariances=None,
                 random_state=None):
        self.n_components = n_components
        self.priors = priors
        self.means = means
        self.covariances = covariances
        self.random_state = check_random_state(random_state)

    def from_samples(self, X, n_iter):
        """EM."""
        n_samples, n_features = X

        if self.priors is None:
            self.priors = np.ones(self.n_components)

        if self.means is None:
            indices = arange(n_samples)
            self.means = X[np.random.choice(indices, self.n_components)]

        if self.covariances is None:
            self.covariances = np.zeros((self.n_samples, n_features,
                                         n_features))

        for i in range(n_iter):
            # Expectation
            r = np.ndarray((self.n_components, n_samples))
            for k in range(self.n_components):
                r[k] = self.priors[k] * MultivariateNormal(
                    mean=self.means[k], covariance=self.covariances[k],
                    random_state=self.random_state).to_probability_density(X)
            r /= r.sum(axis=0)

            # Maximization
            #N = np.sum(responses,axis=1)

            #for i in range(k):
                #mu = np.dot(responses[i,:],data) / N[i]
                #sigma = np.zeros((d,d))

                #for j in range(n):
                   #sigma += responses[i,j] * np.outer(data[j,:] - mu, data[j,:] - mu)

                #sigma = sigma / N[i]

                #self.comps[i].update(mu,sigma) # update the normal with new parameters
                #self.priors[i] = N[i] / np.sum(N) # normalize the new priors
                

    def sample(self, n_samples):
        pass
        # TODO

    def to_probability_density(self, X):
        pass
        # TODO

    def marginalize(self, indices):
        pass
        # TODO

    def condition(self, indices, x):
        pass
        # TODO

    def predict(self, indices, x):
        pass
        # TODO
