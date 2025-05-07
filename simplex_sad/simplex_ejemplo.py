import numpy as np
from tabulate import tabulate

def simplex(A, b, c):
    """
    Resuelve el problema de maximización:
        max c^T x
        sujeto a A x <= b, x >= 0
    mediante el método simplex en forma de tabla.

    Parámetros:
    A: numpy.ndarray de tamaño (m, n) coeficientes de restricciones
    b: numpy.ndarray de tamaño (m,) términos independientes
    c: numpy.ndarray de tamaño (n,) coeficientes de la función objetivo

    Retorna:
    (valor_optimo, x_optimo, tableau)
    valor_optimo: float
    x_optimo: numpy.ndarray de tamaño (n,) solución óptima
    tableau: numpy.ndarray la tabla final usada
    """
    m, n = A.shape
    # Construir tabla inicial (m restricciones + fila Z) y (n variables + m holgura + LD)
    tableau = np.zeros((m+1, n + m + 1), dtype=float)
    # Restricciones
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    # Función objetivo
    tableau[-1, :n] = -c

    # Iteraciones simplex
    while True:
        # Criterio de optimalidad: no hay coeficientes negativos en fila Z
        cost_row = tableau[-1, :-1]
        if np.all(cost_row >= 0):
            break
        # Elegir columna pivote: más negativo en fila Z
        pivot_col = np.argmin(cost_row)
        # Razón mínima para fila pivote
        col = tableau[:m, pivot_col]
        rhs = tableau[:m, -1]
        ratios = np.where(col > 0, rhs / col, np.inf)
        if np.all(ratios == np.inf):
            raise ValueError("Problema no acotado (unbounded)")
        pivot_row = np.argmin(ratios)
        # Normalizar fila pivote
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val
        # Eliminar entradas en la columna pivote para otras filas
        for i in range(m+1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Extraer solución óptima
    x_opt = np.zeros(n)
    for j in range(n):
        col = tableau[:m, j]
        if np.count_nonzero(col == 1) == 1 and np.sum(col) == 1:
            row = np.where(col == 1)[0][0]
            x_opt[j] = tableau[row, -1]
    valor_opt = tableau[-1, -1]
    return valor_opt, x_opt, tableau

# Ejemplo de uso:
if __name__ == '__main__':
    A = np.array([[1,0],[0,2],[3,2],], dtype=float)
    b = np.array([4, 12, 18],dtype=float)
    c = np.array([30000, 50000], dtype=float)
    opt, x_opt, final_table = simplex(A, b, c)
    print(f"Valor óptimo: {opt}")
    print(f"Solución x: {x_opt}")
    print("Tabla final:")
    print(tabulate(final_table, tablefmt='grid', floatfmt='.2f'))
