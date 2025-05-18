
#
#algoritmo de gradiente descendente
#

import derivada as d
import numpy as np 
import sympy as sp

#  x**2+y**2

def Gradiente(variables,funcion,**kwars):
    print(kwars)

    simbolos = sp.symbols((variables))

    funcionOriginal = sp.sympify(funcion)

    # for var in variables:
    #     print("var: ", var)
    #     print("kwars: ", kwars[var][0])

    
    # for var in range(len(variables),len(kwars)):
    #     #print({kwars[var]})
    #     variableDelta = 
    #     print("var: ", kwars[0])


    puntos = [kwars[var][0] for var in variables]
    deltas = [kwars[f'delta{var.upper()}'][0] for var in variables]

    print("puntos: ",puntos)
    print("deltas: ",deltas)


    gradienteSimbolicos = [sp.diff(funcionOriginal,var) for var in variables]
    print("gradientes simbolicos")
    print(gradienteSimbolicos)

    
    print("gradiente siendo evaluado")
    puntosDict = dict(zip(simbolos,puntos))
    # print('ola: ',ola)
    for funcionGradiente in gradienteSimbolicos:
        print(funcionGradiente)
        print(f"{funcionGradiente}{puntos} = ", funcionGradiente.evalf(subs=puntosDict))


    gradienteAplicado = []
    # enumerate(simbolos)
    # print(deltas)
    # print(enumerate(simbolos))
    for i, var in enumerate(simbolos):
        print(i,var)

        puntoDelta = puntos.copy()
        
        puntoDelta[i] += deltas[i]

        # print("puntoDelta: ",puntoDelta)

        print("aplicando formula de la derivada")

        funcionMasDelta = funcionOriginal.evalf(subs=dict(zip(simbolos,puntoDelta)))
        funcionNormal = funcionOriginal.evalf(subs=puntosDict)
        operacion = (funcionMasDelta - funcionNormal) / deltas[i]
        gradienteAplicado.append(operacion)

        print(f"∂f/∂{var} ≈ {operacion}")
        
    # print("Gradiente simbolica")
    # for i in range(0,len(vector)):
    #     derivadaEn = vector[i]
        
    #     funcionDerivada = sp.diff(funcion,derivadaEn)
    
    #     gradienteSimbolica.append(funcionDerivada)
    #     print(f"{funcionDerivada}")

    # print("Gradiente evaluado en un punto")
    # for i in range(0,len(gradienteSimbolica)):

    #     funcion = gradienteSimbolica[i]

    #     for j in range(0,len(xPuntos)):
         
    #         valor = funcion.subs([(x,xPuntos[j]), (y,yPuntos[j])])
    #         print(f"{funcion} ([{xPuntos[j]},{yPuntos[j]}]) = {valor}")
    #         # print(valor)



#Gradiente(['x','y'],"x**2 + y**2",**{'x':[2],'y':[3],'deltaX':[0.01],'deltaY':[0.01]})


Gradiente(['x','y','z'],"x**2 + y**2 + z**2" ,**{'x':[2],'y':[3],'z':[4],'deltaX':[0.01],'deltaY':[0.01],'deltaZ':[0.01]})