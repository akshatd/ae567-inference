import numpy as np
import matplotlib.pyplot as plt

# For comments on these functions see the Gaussian Random Variable Notebook
def normpdf(x, mean, cov):
    n, N = x.shape
    preexp = 1.0 / (2.0 * np.pi)**(n/2) / np.linalg.det(cov)**0.5
    diff = x - np.tile(mean[:, np.newaxis], (1, N))
    sol = np.linalg.solve(cov, diff)
    inexp = np.einsum("ij,ij->j",diff, sol)
    out = preexp * np.exp(-0.5 * inexp)
    return out

def eval_normpdf_on_grid(x, y, mean, cov):
    XX, YY = np.meshgrid(x,y)
    pts = np.stack((XX.reshape(-1), YY.reshape(-1)),axis=0)
    evals = normpdf(pts, mean, cov).reshape(XX.shape)
    return XX, YY, evals

def plot_bivariate_gauss(fignum, x, y, mean, cov):
    std1 = cov[0,0]**0.5
    std2 = cov[1,1]**0.5
    mean1 = mean[0]
    mean2 = mean[1]
    XX, YY, evals = eval_normpdf_on_grid(x, y, mean, cov)
    fig, axis = plt.subplots(2, 2, num=fignum)
    axis[0,0].plot(x, normpdf(x[np.newaxis,:], np.array([mean1]), np.array([[std1**2]])))
    axis[0,0].set_ylabel(r'$f_{\theta_1}$')
    axis[1,1].plot(normpdf(y[np.newaxis,:], np.array([mean2]), np.array([[std2**2]])),y)
    axis[1,1].set_xlabel(r'$f_{\theta_2}$')
    axis[1,0].contourf(XX, YY, evals)
    axis[1,0].set_xlabel(r'$\theta_1$')
    axis[1,0].set_ylabel(r'$\theta_2$')
    axis[0,1].set_visible(False)
    return fig, axis