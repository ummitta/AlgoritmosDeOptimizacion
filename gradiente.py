
import numpy as np
import sympy as sp

#  x**2+y**2

def Gradiente(variables, funcion, **kwars):

    #se crean los simbolos
    simbolos = sp.symbols((variables))

    #se convierte la funcion a expresion simbolica
    funcionOriginal = sp.sympify(funcion)

    #tomamos los puntos y los deltas del kwars
    puntos = [kwars[var][0] for var in variables]
    deltas = [kwars[f'delta{var.upper()}'][0] for var in variables]

    print("puntos: ",puntos)
    print("deltas: ",deltas)

    #se calcula el gradiente simbolico
    gradienteSimbolicos = [sp.diff(funcionOriginal,var) for var in variables]
    print("gradientes simbolicos")
    print(gradienteSimbolicos)

    
    print("gradiente siendo evaluado")
    #se crea un diccionario con los simbolos y sus puntos
    puntosDict = dict(zip(simbolos,puntos))
    # se calcula la gradiente con los puntos dados
    for funcionGradiente in gradienteSimbolicos:
        print(f"{funcionGradiente}{puntos} = ", funcionGradiente.evalf(subs=puntosDict))


    gradienteAplicado = []
    # se calcula la aproximacion numerica del gradiente
    for i, var in enumerate(simbolos):
        print(i,var)

        puntoDelta = puntos.copy()
        
        puntoDelta[i] += deltas[i]

        print("aplicando formula de la derivada")

        funcionMasDelta = funcionOriginal.evalf(subs=dict(zip(simbolos,puntoDelta)))
        funcionNormal = funcionOriginal.evalf(subs=puntosDict)
        operacion = (funcionMasDelta - funcionNormal) / deltas[i]
        gradienteAplicado.append(operacion)

        print(f"∂f/∂{var} ≈ {operacion}")
        

Gradiente(['x','y'],"x**2 + y**2" ,**{'x':[2],'y':[3],'deltaX':[0.01],'deltaY':[0.01]})
print("----------------------------------------------")

Gradiente(['x','y','z'],"x**2 + y**2 + z**2" ,**{'x':[2],'y':[3],'z':[4],'deltaX':[0.01],'deltaY':[0.01],'deltaZ':[0.01]})