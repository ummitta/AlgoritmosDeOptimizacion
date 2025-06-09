
import sympy as sp
import numpy as np

#epsilon = np.finfo(float).eps


def newton(funcion,puntoUsuario,t,epsilon):

    x = sp.symbols('x')

    derivada = sp.diff(funcion,x)

    hessiana = sp.diff(derivada,x)
    print("derivada: ", derivada)
    print("hessiaana: ", hessiana)

    funcionSimbolica = sp.sympify(funcion)

    #print("funcionSimbolica: ", funcionSimbolica)
    punto = puntoUsuario

    verificacion = True

    while verificacion:

        gradientePunto = derivada.subs(x,punto)

        hessianaPunto = hessiana.subs(x,punto)
        #print("derivada: ", derivada)
        #print("hessiaana: ", hessiana)

        if hessianaPunto == 0:
            print("Nose se puede divir por 0")

        puntoNuevo = punto - t * (gradientePunto/hessianaPunto)

        #print(f"punto: {punto} | gradiente: {gradientePunto} | hessiana: {hessianaPunto} | puntoNuevo: {puntoNuevo}  ")
        print(f" hessiana: {hessianaPunto} ")

        print(f"punto: {punto} | gradiente: {gradientePunto} | hessiana: {float(hessianaPunto):.4f}  | puntoNuevo: {puntoNuevo}  ")

        if sp.Abs(puntoNuevo - punto) <= epsilon:
            #print("se rompeee")
            #print(f"punto: {punto} | gradiente: {gradientePunto} | hessiana: {hessianaPunto} | puntoNuevo: {puntoNuevo}  ")

            verificacion = False
            continue
        #puntoNuevo
        punto = puntoNuevo

    return float(puntoNuevo)

epsilon = np.finfo(float).eps

punto = newton("x**3 - x - 2",2,1,epsilon)

print(punto)
