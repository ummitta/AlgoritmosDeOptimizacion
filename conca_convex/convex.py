import math
import numpy as np
import matplotlib.pyplot as plt

# -(x - 1.5)**2 + 1.5    [0.8, 2.2]
#  (x - 1)**2 + 1        [0.5, 1.5]
#  x**3                  [-3 , 3  ]

def e(x, funct):
    f = funct.split(' ')
    for i in range(len(f)):
        if f[i] == 'x':
            f[i] = x
    FuncionFinal = ''        
    for e in f:
        FuncionFinal += str(e)
    return eval(FuncionFinal)

def graficar(f, x0, x1, lam, la_funcion, xmin=-3, xmax=3 ):

    x_lambda = lam * x1 + (1 - lam) * x0
    f_x0 = f(x0)
    f_x1 = f(x1)
    f_xlambda = f(x_lambda)
    convex_comb = lam * f_x1 + (1 - lam) * f_x0

    x = np.linspace(xmin, xmax, 400)
    y = f(x)


    # graficando

    plt.figure(figsize=(8, 6))

    plt.plot(x, y, 'k', label='f(x)')

    plt.plot([x0, x1], [f_x0, f_x1], 'ro')
    
    z = np.linspace(x0, x1, 400)
    w = f(z)
    plt.plot(z, w, 'b')
    
    plt.plot([x0, x1], [f_x0, f_x1], 'r--', label=r'$\lambda f(x_1) + (1-\lambda) f(x_0)$')

    plt.plot(x0, f_x0, 'ro', label='x0')
    plt.plot(x1, f_x1, 'ro', label='x1')

    plt.vlines(x0, 0, f_x0, colors='green', linestyles=':', label=r'$f(\lambda x_1 + (1-\lambda)x_0)$')
    plt.vlines(x1, 0, f_x1, colors='green', linestyles=':')

    plt.title(f"LA FUNCION: {la_funcion}")

    # eje x e y
    plt.axhline(0,color="black",linewidth=1)
    plt.axvline(0,color="black",linewidth=1)
    # etiquetas eje x e y
    plt.xlabel('x')
    plt.ylabel('f(x)')

    # malla de fondo o grid
    plt.grid(True)

    plt.legend()

    # limites para el cual se graficara la funcion
    plt.ylim(0, max(f_x0, f_x1, f_xlambda) + 1)
    plt.xlim(xmin, xmax)
    plt.show()

FuncionIn = str(input("ingrese funcion a evaluar\n"))

xa = float(input('ingrese el valor de Xa\n'))

xb = float(input('ingrese el valor de Xb\n'))

dlambda = float(input('ingrese el valor de delta lambda\n'))

array = np.arange(0, 1, dlambda, dtype=int)

array = np.linspace(0, 1, len(array), dtype=float)


concavo = 0
convexo = 0
i = 0

while i < len(array):
    lamda = array[i]
    
    if (e(lamda * xa + (1 - lamda) * xb, FuncionIn)) < (lamda * e(xa, FuncionIn) + (1 - lamda) * e(xb, FuncionIn)):
        convexo += 1
    if (e(lamda * xa + (1 - lamda) * xb, FuncionIn)) > (lamda * e(xa, FuncionIn) + (1 - lamda) * e(xb, FuncionIn)):
        concavo += 1
    
    i += 1

if convexo == len(array)-2:
    print("la funcion es convexa")
    la_funcion = "es convexa"

elif concavo == len(array)-2:
    print("la funcion es concava")
    la_funcion = "es concava"

else:
    print("la funcion no es ni concava ni convexa.")
    print(concavo, convexo)
    la_funcion = "no es ni concava ni convexa"

# Función convexa (cuadrática)
graficar(lambda x: eval(FuncionIn), xa, xb, dlambda, la_funcion, xmax=3)
