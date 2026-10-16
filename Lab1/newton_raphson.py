import numpy as np

# f(x,y) = x**4 + 2*y**4 + x**2*y**2 - 6*x - 10*y
def grad(u):
    x, y = u
    return np.array([
        4*x**3 + 2*x*y**2 - 6, 
        8*y**3 + 2*x**2*y - 10 
    ])


def hessian(u):
    x, y = u
    return np.array([
        [
            12*x**2 + 2*y**2,
            4*x*y
        ],

        [
            4*x*y,
            24*y**2 + 2*x**2
        ]
    ])


def newton_optim(u_0, grad_fn, hessian_fn, epsilon = 10e-8, max_iter = 100):
    u = u_0
    for _ in range(max_iter):
        grad = grad_fn(u)
        if np.linalg.norm(grad) < epsilon:
            return u
        hessian = hessian_fn(u)
        t = np.linalg.solve(hessian, -grad)
        u = u + t
    return u

if __name__ == "__main__":
    u_0 = (10, -10)
    u_optim = newton_optim(u_0, grad, hessian, max_iter=1000000)
    print(u_optim)