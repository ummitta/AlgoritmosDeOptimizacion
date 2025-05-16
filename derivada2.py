#
# Codigo De Derivada Numerica para resolucion de problemas de Programacion No Lineal
# por Nicolas Barros D.
#

import derivada as d
from sympy import diff, Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy import sympify
from sympy import limit
import numpy as np
import sympy as sp

f = input('Ingrese Funcion a calcular\n')
x = input('Derivar esta funcion sobre: ')
xvalor = float(input('Valor de x: '))
deltax = float(input('Valor de deltax: '))

derivada = d.pDeriv(f, x)
der = derivada.subs(x, 2)

print('La derivada de la funcion es: ')
print(derivada.subs(x, 2))

limitexd = d.limitex(f, x)
fx_dx = limitexd.subs(x, xvalor + deltax)
fx = limitexd.subs(x, xvalor)
print(fx_dx)
print(fx)
print(deltax)
limite = (fx_dx - fx)/deltax
#
print('El limite es: ')
print(abs(der - limite))
if abs(der - limite) < 1e-16:
    print("olakase")

# lim deltax->0: f(x + deltax) - f(x)/deltax 

#x**3 + x + 1