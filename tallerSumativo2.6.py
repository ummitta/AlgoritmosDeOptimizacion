# Taller Sumativo 2.6: Matriz hessiana y teorema de Taylor.
# Autores: Nicolas Barros, Maximo Mora

import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

#los vectores -> son de n x 1

#matriz -> son de m x n

#delta -> como columna


def TeoremaTaylor(funcion,intervalo,delta,t):
    x = sp.symbols('x')

    funcionSimbolica = sp.simplify(funcion)

    funcionEvaluada = funcionSimbolica.subs({x:intervalo[1]})

    deltaMatrix = np.array(delta)

    print("funcionEvaluada: ", funcionEvaluada)

    derivadaX = sp.diff(funcionSimbolica,x)

    print("derivadaX: ",derivadaX)
    print("deltaMatrix: ", deltaMatrix)

    matrixHessiana = sp.diff(derivadaX,x)

    print("matrixHessiana: ", matrixHessiana)
    print("deltaMatrix: ", deltaMatrix)
    print(f"{funcionEvaluada } + {derivadaX} * {deltaMatrix} + 1 / 2 * {deltaMatrix} * {matrixHessiana} * {deltaMatrix}")

    funcionAproximada = funcionSimbolica + derivadaX * deltaMatrix + (1/2) * deltaMatrix * matrixHessiana * deltaMatrix

    funcionAproximadaSimbolica = sp.simplify(funcionAproximada)

    print("funcionAproximadaSimbolica: ",funcionAproximadaSimbolica)

    xF = np.linspace(-10, 10, 100)  

    funcionNormalLambificada = sp.lambdify(x, funcionSimbolica, 'numpy')
    y1 = funcionNormalLambificada(xF)
    funcionAproximadaLambificada = sp.lambdify(x, funcionAproximadaSimbolica, 'numpy')
    y2 = funcionAproximadaLambificada(xF)

    plt.plot(xF, y1, 'r' , label=f'f1(x) = {funcion}')
    plt.plot(xF, y2, 'b' ,label=f'f2(x) = {funcionAproximadaSimbolica}')

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)  

    plt.title('Teorema de Taylor')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

TeoremaTaylor("-x + cos(x)",[0,2],0.1,1)
