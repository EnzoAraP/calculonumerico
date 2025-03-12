
import numpy as np

def compute_gj(j, a, b, m=1000):
    """Compute the integral of x^(j-1) from a to b using composite Simpson's rule."""
    h = (b - a) / m
    integral = 0.0
    for k in range(m + 1):
        x = a + k * h
        if k == 0 or k == m:
            coeff = 1
        elif k % 2 == 1:
            coeff = 4
        else:
            coeff = 2
        integral += coeff * (x ** (j - 1))
    integral *= h / 3
    return integral

def initial_guess(a, b, N):
    """Generate initial guesses for weights w and nodes t."""
    w = np.zeros(N)
    t = np.zeros(N)
    # Compute initial weights
    for i in range(N):
        if i < N // 2:
            w[i] = (b - a) / (2 * N) * (i + 1)
        else:
            w[i] = w[N - 1 - i]
    # Compute initial nodes
    if N % 2 == 0:
        for i in range(N):
            if i < N // 2:
                t[i] = a + (i + 1) * w[i] / 2
            else:
                t[i] = (a + b) - t[N - 1 - i]
    else:
        middle = (N - 1) // 2
        t[middle] = (a + b) / 2
        for i in range(N):
            if i < middle:
                t[i] = a + (i + 1) * w[i] / 2
            elif i > middle:
                t[i] = (a + b) - t[N - 1 - i]
    return w, t

def compute_f(w, t, g):
    """Compute the residual vector f."""
    N = len(w)
    f = np.zeros(2 * N)
    for j in range(1, 2 * N + 1):
        fj = 0.0
        for i in range(N):
            fj += w[i] * (t[i] ** (j - 1))
        fj -= g[j - 1]  # g is 0-based
        f[j - 1] = fj
    return f

def compute_jacobian(w, t, g, epsilon=1e-8):
    """Approximate the Jacobian matrix using finite differences."""
    N = len(w)
    size = 2 * N
    J = np.zeros((size, size))
    f_current = compute_f(w, t, g)
    # Perturb each variable and compute the difference
    for col in range(size):
        w_pert = np.copy(w)
        t_pert = np.copy(t)
        if col < N:
            # Perturb w[col]
            w_pert[col] += epsilon
        else:
            # Perturb t[col - N]
            t_pert[col - N] += epsilon
        f_perturbed = compute_f(w_pert, t_pert, g)
        J[:, col] = (f_perturbed - f_current) / epsilon
    return J

def newton_method(a, b, N, tol=1e-8, max_iter=100):
    """Perform Newton-Raphson iteration to find weights and nodes."""
    w, t = initial_guess(a, b, N)
    # Compute the vector g of integrals
    g = np.array([compute_gj(j, a, b) for j in range(1, 2 * N + 1)])
    # Newton iteration
    for iteration in range(max_iter):
        f = compute_f(w, t, g)
        residual = np.max(np.abs(f))
        print(f"Iteration {iteration + 1}: Residual = {residual:.2e}")
        if residual < tol:
            print(f"Converged after {iteration + 1} iterations.")
            return w, t
        # Compute Jacobian
        J = compute_jacobian(w, t, g)
        # Solve J * s = -f
        try:
            s = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            print("Jacobian is singular. Stopping iteration.")
            return w, t
        # Update w and t
        delta_w = s[:N]
        delta_t = s[N:]
        w += delta_w
        t += delta_t
    print("Maximum iterations reached without convergence.")
    return w, t

# Example usage
a, b = -1, 1
N = 2
print(f"Computing Gauss-Legendre quadrature for interval [{a}, {b}] with N={N} points.")
weights, nodes = newton_method(a, b, N)
print("\nComputed weights:", weights)
print("Computed nodes:", nodes)