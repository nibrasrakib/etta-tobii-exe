from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.manifold import _t_sne
import numpy as np
from time import time
from scipy import linalg


'''
Wrapper around TSNE to output the internal data
'''

# print out internal data and final embedded data


def main():
    X = np.array([[0, 0, 0, 4], [0, 1, 1, 5], [1, 0, 1, 3],
                  [1, 1, 1, 2], [3, 1, 2, 2]])
    tsne = MyTSNE()
    X_embedded = tsne.fit_transform(X)

    print(tsne.coordinates)
    print(X_embedded)


# extends TSNE

class MyTSNE(TSNE):
    def __init__(self, n_components=2, perplexity=30.0,
                 early_exaggeration=100.0, learning_rate=200.0, n_iter=5000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5):

        super(MyTSNE, self).__init__(
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
            n_iter_without_progress=n_iter_without_progress,
            min_grad_norm=min_grad_norm,
            metric=metric,
            init=init,
            verbose=verbose,
            random_state=random_state,
            method=method,
            angle=angle)

        # additional variable
        self.coordinates = []
        self._EXPLORATION_N_ITER = 250  # Define the number of exploration iterations

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded,
              neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        if self.method == 'barnes_hut':
            obj_func = manifold._t_sne._kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
        else:
            obj_func = manifold.t_sne._kl_divergence

        # Learning schedule (part 1): do 250 iterations with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it, _ = MyTSNE._gradient_descent(
            obj_func, params, **opt_args)

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = \
                self.n_iter_without_progress
            params, kl_divergence, it, p_list = \
                MyTSNE._gradient_descent(obj_func, params, **opt_args)
            coordinates = np.array(p_list)
            self.coordinates = coordinates.reshape(
                -1, n_samples, self.n_components)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    @staticmethod
    def _gradient_descent(objective, p0, it, n_iter,
                          n_iter_check=1, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
        """Batch gradient descent with momentum and individual gains."""
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        p_list = []

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float64).max
        best_error = np.finfo(np.float64).max
        best_iter = i = it

        tic = time()
        for i in range(it, n_iter):
            check_convergence = (i + 1) % n_iter_check == 0
            # only compute the error when needed
            kwargs['compute_error'] = \
                check_convergence or i == n_iter - 1

            error, grad = objective(p, *args, **kwargs)
            grad_norm = linalg.norm(grad)

            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

            # get intermediate coordinates
            threshold = 20.0
            if len(p_list) == 0:
                p_list.extend(p)
            else:
                # add xy coordinates if at least one point moves more
                # than threshold
                if max(abs(np.array(p_list[-len(p):]) - p)) > threshold:
                    p_list.extend(p)

            if check_convergence:
                toc = time()
                duration = toc - tic
                tic = toc

                if verbose >= 2:
                    print("[t-SNE] Iteration %d: error = %.7f,"
                          " gradient norm = %.7f"
                          " (%s iterations in %0.3fs)"
                          % (i + 1, error, grad_norm,
                             n_iter_check, duration))

                if error < best_error:
                    best_error = error
                    best_iter = i
                elif i - best_iter > n_iter_without_progress:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: did not "
                              "make any progress "
                              "during the last %d episodes. Finished."
                              % (i + 1, n_iter_without_progress))
                    break
                if grad_norm <= min_grad_norm:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: gradient "
                              "norm %f. Finished."
                              % (i + 1, grad_norm))
                    break

        print("Extracted", len(p_list) / 20, "coordinates")
        return p, error, i, p_list


if __name__ == "__main__":
    main()
