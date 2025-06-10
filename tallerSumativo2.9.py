# Taller Sumativo 2.9: Algoritmo del gradiente descendente con actualización de t por el método de Barzilai-Borwein
# Autores: Nicolas Barros, Maximo Mora

import sympy as sp
import numpy as np
import math as ma
import matplotlib.pyplot as plt

def normal(xy):
    return np.linalg.norm(xy)

def reporte_resultados(resultado, e, t, xys):
    print("Gradiente Descendente con Barzilai-Borwein")
    print("-------------------------------------------")
    print("Función:", func_grad)
    print("Punto inicial:", xys)
    print("Epsilon:", e)
    print("Resultado x e y:   ", resultado[0])
    print("Gradiente Evaluado:", resultado[1])
    print("Iteraciones:", len(resultado[2]) - 1)

def grad_des(funcion, xy, t, epsilon):
    lista = np.array([xy])
    x, y = sp.symbols('x y')

    grad = [sp.diff(funcion, x), sp.diff(funcion, y)]

    g = np.array([
        grad[0].subs({x: xy[0], y: xy[1]}),
        grad[1].subs({x: xy[0], y: xy[1]})
    ], dtype=float)

    x_old = None
    g_old = None

    while normal(g) >= epsilon:
        delta = -1 * g
        nuevox = xy + t * delta

        g_new = np.array([
            grad[0].subs({x: nuevox[0], y: nuevox[1]}),
            grad[1].subs({x: nuevox[0], y: nuevox[1]})
        ], dtype=float)

        if x_old is not None:
            s = xy - x_old
            y_vec = g - g_old
            denom = np.dot(y_vec, y_vec)
            if denom != 0:
                t = abs(np.dot(s, y_vec)) / denom
                print(f"t actualizado: {t}")
            else:
                print("Denominador cero, t no actualizado")

        lista = np.vstack((lista, nuevox))
        x_old = xy
        g_old = g
        xy = nuevox
        g = g_new

    return xy, g, lista

def graficar_trayectoria(funcion, trayectoria):
    x_vals = np.linspace(-3, 4, 400)
    y_vals = np.linspace(-3, 4, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    x, y = sp.symbols('x y')
    f_lambd = sp.lambdify((x, y), funcion, modules=['numpy'])
    Z = f_lambd(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.plot(trayectoria[:, 0], trayectoria[:, 1], marker='o', color='red', label='Trayectoria')
    plt.scatter(trayectoria[-1, 0], trayectoria[-1, 1], color='blue', label='Mínimo encontrado')
    plt.title('Gradiente descendente con actualización de t por el método de Barzilai-Borwein')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Entrada de usuario
x, y = sp.symbols('x y')
func_grad = input("Ingrese la función en términos de x e y (ejemplo: 'x**2 + y**2'): ")
func_grad = sp.sympify(func_grad)

xys = np.array([
    float(input("Ingrese el valor inicial de x: ")),
    float(input("Ingrese el valor inicial de y: "))
])

e = float(input("Ingrese el valor de epsilon: "))
t = float(input("Ingrese la tasa de aprendizaje inicial (ej. 0.1): "))

resultado = grad_des(func_grad, xys, t, e)

reporte_resultados(resultado, e, t, xys)
graficar_trayectoria(func_grad, resultado[2])
