import math
import numpy as np

def f(x, funct):
    f = funct.split(' ')
    for i in range(len(f)):
        if f[i] == 'x':
            f[i] = x
    FuncionFinal = ''        
    for e in f:
        FuncionFinal += str(e)
    return eval(FuncionFinal)

FuncionIn = str(input("ingrese funcion a evaluar\n"))

xa = float(input('ingrese el valor de Xa\n'))

xb = float(input('ingrese el valor de Xb\n'))

dlambda = float(input('ingrese el valor de delta lambda\n'))

if dlambda > 0.09: 
    decimales = 1
elif dlambda < 0.09 and dlambda > 0.009:
    decimales = 2
elif dlambda < 0.009 and dlambda > 0.0009:
    decimales = 3
else: decimales = 4

array = np.arange(0, 1, dlambda, dtype=int)

array = np.linspace(0, 1, len(array) + 1, dtype=float)

array = np.around(array, decimals = decimales)


concavo = 0
convexo = 0
i = 0

while i < len(array):
    lamda = array[i]
    if (f(lamda * xa + (1 - lamda) * xb, FuncionIn)) <= (lamda * f(xa, FuncionIn) + (1 - lamda) * f(xb, FuncionIn)):
        convexo += 1

    else:
        concavo += 1
    
    i += 1

if convexo == len(array):
    print("la funcion es convexa")

elif concavo == len(array):
    print("la funcion es concava")

else:
    print("la funcion no es ni concava ni convexa.")
    print(concavo, convexo)
