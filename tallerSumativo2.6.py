import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

#los vectores -> son de n x 1

#matriz -> son de m x n

#delta -> como columna


def TeoremaTaylor(funcion,intervalo,delta,t):


    x,y,z= sp.symbols('x y z')

    funcionSimbolica = sp.simplify(funcion)

    funcionEvaluada = funcionSimbolica.subs({x:intervalo[0][0]})

    deltaMatrix = np.array(delta)

    print("funcionEvaluada: ", funcionEvaluada)

    

    derivadaX = sp.diff(funcionSimbolica,x)
    print("derivadaX: ",derivadaX)

    print("deltaMatrix: ", deltaMatrix)

    cantidadSimbolos = 1
    matrixHessiana =  np.zeros((cantidadSimbolos,cantidadSimbolos), dtype=object)

    for i in range(cantidadSimbolos):
        matrixHessiana[0, 0] = derivadaX

    print("matrixHessiana: ", matrixHessiana)

    print("deltaMatrix: ", deltaMatrix)
    

    print("Formula: ")
    print(f"f(x) +  ▽ f(x)**t : ")

    print(f"{funcionEvaluada } + {derivadaX} * {deltaMatrix} + 1 / 2 * {deltaMatrix} * {matrixHessiana} * {deltaMatrix}")





def Graficar():

    print("graficar")



# TeoremaTaylor("x**2",[[0],[2]],[0.1],1)

# TeoremaTaylor("x**2 + y**2",[[0,0],[0,2]],[0.1],1)