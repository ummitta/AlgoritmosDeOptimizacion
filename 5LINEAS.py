import sympy as sp
import numpy as np 

import math as ma

def normal(xy):
    return ma.sqrt(xy[0]**2 + xy[1]**2)

def grad_des(funcion, xy, t, epsilon):

    x, y = sp.symbols('x y')
    
    gradiente = [sp.diff(funcion,x), sp.diff(funcion,y)]
    gradienteEvaluado = np.array([gradiente[0].subs({x: xy[0]}), gradiente[1].subs({y: xy[1]})])

    print("gradiente ", gradiente)
    print("gradienteEvaluado ", gradienteEvaluado)

    print("normal(gradienteEvaluado) ", normal(gradienteEvaluado))
    while normal(gradienteEvaluado) >=  epsilon:
        delta = -1 * gradienteEvaluado
        print(delta)
        nuevox = xy + t * delta
        print("nuevox ", nuevox)
        gradienteEvaluado = np.array([gradiente[0].subs({x: nuevox[0]}), gradiente[1].subs({y: nuevox[1]})])
        print(gradienteEvaluado)

        xy = nuevox


    return gradienteEvaluado

grad_des("x**2 + y**2",np.array([1,1]), np.array([0.1]), 10e-6)