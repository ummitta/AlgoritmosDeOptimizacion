
#
#algoritmo de gradiente descendente
#

import derivada as d
import numpy as np 
import sympy as sp

#  x**2+y**2

def Gradiente(vector,funcion,**kwars):
    print(kwars)

    x, y, z = sp.symbols('x y z')

    funcion = sp.sympify(funcion)

    xPuntos = kwars.get('x')
    yPuntos = kwars.get('y')
    xDelta = kwars.get('deltaX')
    yDelta = kwars.get('deltaY')
    # print(xPuntos)
    # print(yPuntos)
    # print(xDelta)
    # print(yDelta)



    gradienteSimbolica = []
    gradienteAplicada = []
    print("Gradiente simbolica")
    for i in range(0,len(vector)):
        derivadaEn = vector[i]
        
        funcionDerivada = sp.diff(funcion,derivadaEn)
    
        gradienteSimbolica.append(funcionDerivada)
        print(f"{funcionDerivada}")

    print("Gradiente evaluado en un punto")
    for i in range(0,len(gradienteSimbolica)):

        funcion = gradienteSimbolica[i]

        for j in range(0,len(xPuntos)):
         
            valor = funcion.subs([(x,xPuntos[j]), (y,yPuntos[j])])
            print(f"{funcion} ([{xPuntos[j]},{yPuntos[j]}]) = {valor}")
            # print(valor)

    
    print("Aproximacion ")
    for i in range(0,len(gradienteSimbolica)):

        funcion = gradienteSimbolica[i]

        for j in range(0,len(xPuntos)):

            xPunto = xPuntos[j] + xDelta[j]
            yPunto = yPuntos[j] + yDelta[j]
            
     
            valor = funcion.subs([(x,xPunto), (y,yPunto)])
            print(f"{funcion} ([{xPuntos[j]},{yPuntos[j]}]) = {valor}")
          





Gradiente(['x','y'],"x**2 + y**2",**{'x':[2],'y':[3],'deltaX':[0.01],'deltaY':[0.01]})