import numpy as np
from dataclasses import dataclass

def analytic_solution_dirichlet(t, x, viscosity):
    """Analytic solution for dirichlet problem.

    Obtained from: Decomposition method for Solving Burgers’ Equation with Dirichlet and Neumann boundary conditions, Optik, 2017.
     """
    out = 2 * viscosity * np.pi * np.exp(- np.pi**2 * viscosity * t) * np.sin(np.pi * x) / \
            (1.0 + 1e-10 + np.exp(- np.pi**2 * viscosity * t) * np.cos(np.pi * x))
    return out

def get_ic_dirichlet(x, viscosity):
    """Initial condition of dirichlet solution."""
    return analytic_solution_dirichlet(0.0, x, viscosity)

def solve(dt: float,
          dx: float,
          num_steps: int,
          initial_condition: np.ndarray,
          viscosity: float):
    """Solve Burgers equation on x \\in [0,1].

    Note: Assume dirichlet boundary conditions. initial_condition
    should respect them.
    """

    n = initial_condition.shape[0]

    # u has shape (n,)
    # interior nodes are 1:-1
    sol = np.zeros((num_steps, n))
    sol[0, :] = initial_condition
    u = sol[0, :]
    for ii in range(1, num_steps):

        diff = viscosity * (u[2:n] - 2 * u[1:n-1] + u[:n-2]) / dx**2

        # upwind scheme https://en.wikipedia.org/wiki/Upwind_scheme
        # not efficient implementation (purposely slow)
        un = np.zeros(n)
        for jj in range(1, n-1):
           # when u > 0.0, the upwind direction is to the left,
           # and downwind to the right. Which means update should depend on
           # current point, and the point to the left
            if u[jj] > 0:
                un[jj] = u[jj] * (u[jj] - u[jj-1]) / dx
            else:
                un[jj] = u[jj] * (u[jj+1] - u[jj]) / dx

        # assumes dirichlet boundary conditions
        sol[ii, 1:-1] =  u[1:-1] + dt * (diff - un[1:-1])
        u = sol[ii, :]

    return sol