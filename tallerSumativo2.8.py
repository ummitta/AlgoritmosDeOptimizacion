import sympy as sp
import numpy as np 
import math as ma

def normal(xy):
    return ma.sqrt(xy[0]**2 + xy[1]**2)

def reporte_resultados(resultado, e, t, xys):
    print("Gradiente Descendente")
    print("-----------------------")
    print("Función: x^2 + y^2")
    print("Punto inicial:", xys)
    print("Tasa de aprendizaje:", t)
    print("Epsilon:", e)
    print("Resultado x e y:   ", resultado[0])
    print("Gradiente Evaluado:", resultado[1])
    print("Iteraciones:", len(resultado[2]))

def grad_des(funcion, xy, t, epsilon):
    lista = np.array(xy)
    x, y = sp.symbols('x y')

    gradiente = [sp.diff(funcion,x), sp.diff(funcion,y)]
    gradienteEvaluado = np.array([gradiente[0].subs({x: xy[0]}), gradiente[1].subs({y: xy[1]})])

    while normal(gradienteEvaluado) >=  epsilon:
        delta = -1 * gradienteEvaluado
        nuevox = xy + t * delta
        gradienteEvaluado = np.array([gradiente[0].subs({x: nuevox[0]}), gradiente[1].subs({y: nuevox[1]})])
        lista = np.vstack((lista, nuevox))
        xy = nuevox

    return xy, gradienteEvaluado, lista

x, y = sp.symbols('x y')

func_grad = input("Ingrese la función en términos de x e y (ejemplo: 'x**2 + y**2'): ")
func_grad = sp.sympify(func_grad)

xys = np.array([float(input("Ingrese el valor inicial de x: ")), float(input("Ingrese el valor inicial de y: "))])

e = float(input("Ingrese el valor de epsilon: "))
t = float(input("Ingrese la tasa de aprendizaje: "))

resultado = grad_des(func_grad, xys, np.array([t]), e)

reporte_resultados(resultado, e, t, xys)