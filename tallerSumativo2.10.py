# Taller Sumativo 2.10: Método de Newton
# Autores: Nicolas Barros, Maximo Mora

import sympy as sp
import numpy as np

def newton(funcion,puntoUsuario,t,epsilon):

    x = sp.symbols('x')

    derivada = sp.diff(funcion,x)

    hessiana = sp.diff(derivada,x)
    print("derivada: ", derivada)
    print("hessiaana: ", hessiana)

    funcionSimbolica = sp.sympify(funcion)

    punto = puntoUsuario

    verificacion = True

    while verificacion:

        gradientePunto = derivada.subs(x,punto)
        hessianaPunto = hessiana.subs(x,punto)

        if hessianaPunto == 0:
            print("Nose se puede divir por 0")

        puntoNuevo = punto - t * (gradientePunto/hessianaPunto)

        print(f" hessiana: {hessianaPunto} ")
        print(f"punto: {punto} | gradiente: {gradientePunto} | hessiana: {float(hessianaPunto):.4f}  | puntoNuevo: {puntoNuevo}  ")

        if sp.Abs(puntoNuevo - punto) <= epsilon:
            verificacion = False
            continue
        
        punto = puntoNuevo

    return float(puntoNuevo)

epsilon = np.finfo(float).eps

punto = newton("x**3 - x - 2",2,1,epsilon)

print(punto)


#  ¿Cuál es el problema más importante en el algoritmo para el método de Newton con múltiples variables?
#
# Para responder esta pregunta, podemos ver que nuestro codigo, ocupamos la derivada como un numero. ya que solamente trabajamos con funciones de una variable

# Pero si la funcion tiene mas varaibales, la hessiana no es un numero, sino que una matriz, en cada paso del meotodo, es necesario invertir esta matriz,
# lo cual tiene un costo computacional n^3, lo cual significa que va a mas tiempo y mas recursos, y no siempre se puede invertir: si la matriz no tiene inversa, el método no puede avanzar.
