import math
import numpy as np
import matplotlib.pyplot as plt

def e(x, funct):
    f = funct.split(' ')
    for i in range(len(f)):
        if f[i] == 'x':
            f[i] = x
    FuncionFinal = ''        
    for e in f:
        FuncionFinal += str(e)
    return eval(FuncionFinal)

def graficar(f, x0, x1, lam, hola, xmin=0, xmax=3 ):

    # Punto intermedio y valores correspondientes
    x_lambda = lam * x1 + (1 - lam) * x0
    f_x0 = f(x0)
    f_x1 = f(x1)
    f_xlambda = f(x_lambda)
    convex_comb = lam * f_x1 + (1 - lam) * f_x0

    # Valores para graficar la función
    x = np.linspace(xmin, xmax, 400)
    y = f(x)

    # Crear gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'k', label='f(x)')

    # Puntos en extremos y punto intermedio
    plt.plot([x0, x1], [f_x0, f_x1], 'ro')  # extremos
    plt.plot(x_lambda, f_xlambda, 'bo')     # valor real
    plt.plot(x_lambda, convex_comb, 'go')   # combinación convexa

    # Línea del segmento lineal (convex combination)
    plt.plot([x0, x1], [f_x0, f_x1], 'r--', label=r'$\lambda f(x_1) + (1-\lambda) f(x_0)$')

    # Línea vertical al valor real
    plt.vlines(x_lambda, 0, f_xlambda, colors='blue', linestyles=':', 
               label=r'$f(\lambda x_1 + (1-\lambda)x_0)$')

    # Anotaciones

    plt.title(f"LA FUNCION: {hola}")
    #eje x e y
    plt.axhline(0,color="black",linewidth=1)
    plt.axvline(0,color="black",linewidth=1)


    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
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
    hola = "es convexa"

elif concavo == len(array)-2:
    print("la funcion es concava")
    hola = "es concava"

else:
    print("la funcion no es ni concava ni convexa.")
    print(concavo, convexo)
    hola = "no es ni concava ni convexa"

# Función convexa (cuadrática)
graficar(lambda x: eval(FuncionIn), xa, xb, dlambda, hola, xmax=3)
