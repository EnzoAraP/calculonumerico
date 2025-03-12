import numpy as np


def valores(a,b,N):
    w = np.zeros(N)
    t = np.zeros(N)
    for i in range(N):
        if i <= N/2:
            w[i] = ((b - a) / (2 * N)) * (i + 1)  
        else:
            w[i] = w[N - 1 - i]  
    
    for i in range(N):
        if N % 2 == 0:
            if i < N/2:
                t[i] = a + (i + 1) * w[i] / 2
            else:
                t[i] = (a + b) - t[N - 1 - i]  
        else:
            meio = (N - 1) // 2
            t[meio] = (a + b) / 2
            for i in range(N):
                if i < meio:
                    t[i] = a + (i + 1) * w[i] / 2
                elif i > meio:
                    t[i] = (a + b) - t[N - 1 - i]
    return w, t
def Pontomedio(j,a,b,m=1000):
    h = (b - a) / m
    integral = 0.0
    for k in range(m):
        x_medio = a + (k + 0.5) * h  
        integral += (x_medio) ** (j - 1)
    integral *= h
    return integral
def f(w, t, g):
    N = len(w)
    f = np.zeros(2 * N)
    for j in range(1, 2 * N + 1):
        fj = sum(w[i] * (t[i] ** (j - 1)) for i in range(N)) - g[j - 1]
        f[j - 1] = fj
    return f
def compute_jacobian(w, t, g, epsilon=1e-8):    
    N = len(w)  
    tamanho = 2 * N  
    J = np.zeros((tamanho, tamanho))  
    foriginal = f(w, t, g) 
    
    for col in range(tamanho):
        wdoido = np.copy(w)
        tdoido = np.copy(t)
        if col < N:
            wdoido[col] += epsilon
        else:
            tdoido[col - N] += epsilon
        fdoido = f(wdoido, tdoido, g) 
        J[:, col] = (fdoido - foriginal) / epsilon
    return J
def newton_method(a, b, N, tol=1e-8, max_iter=100):
    g = np.array([Pontomedio(j, a, b) for j in range(1, 2 * N + 1)])
    w, t = valores(a, b, N)

    for iteration in range(max_iter):
        f_vec = f(w, t, g)
        residual = np.max(np.abs(f_vec))
        if residual < tol:
            print(f"Convergência em {iteration+1} iterações. Resíduo: {residual:.2e}")
            return w, t
        
        J = compute_jacobian(w, t, g)
        
        try:
            delta = np.linalg.solve(J, -f_vec) 
        except np.linalg.LinAlgError:
            print("Erro: Jacobiana singular.")
            return w, t
        
        w += delta[:N]
        t += delta[N:]

    print(f"Não convergiu após {max_iter} iterações. Resíduo: {residual:.2e}")
    return w, t

a =-1;
b=1;
N=4;
weights, nodes = newton_method(a, b, N)
print("\nResultados:")
print("Pesos:", np.round(weights, 6))
print("Nós:  ", np.round(nodes, 6))



        

