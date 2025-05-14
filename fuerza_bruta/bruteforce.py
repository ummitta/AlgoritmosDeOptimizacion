
import numpy as np

def f(x1, x2, theta):
    z = (1.2 * x1) + (1.16 * x2) - (theta * ((2 * x1**2) + x2**2 + (x1 + x2)**2))
    return z

tita = float(input('Ingrese el valor de theta\n'))
# [0,1]

deltax = float(input('Ingrese el valor de delta x\n'))
# [0,1]

array = np.arange(0, 5, deltax, dtype=float)

tuplas = []

for i in range (len(array)):
        for j in range(len(array)):

            valmax = f(array[i], array[j],tita)

            if array[i]+array[j]<=5.0:
                tuplas.append((array[i],array[j],valmax))

var1= 0.0

tUpla = ()

for i in range(len(tuplas)):
    print(tuplas[i])
    var2 = tuplas[i][2]

    if var2 > var1:

        tUpla = tuplas[i]
        var1 = var2


print('El largo del arreglo recorrido es: ' + str(len(array)) +':\n')

print('Los valores posibles son los siguientes '+ str(len(tuplas)) +':\n')

print('El valor mayor es '+ str(var1) + ' y se consigue cuando x1 es '+ str(tUpla[0])+ ' y cuando x2 es '+ str(tUpla[1]))

# ¿Por que no guardar el maximo dentro de los ciclos?
#
# Porque el valor mayor se conseguirá despues de evaluar todos los posibles valores 
#
# ¿De que depende la complejidad de la solucion del problema?
#
# La complejidad de este algoritmo es O(n^2) siendo n la cantidad de valores a recorrer