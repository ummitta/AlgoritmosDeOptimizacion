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
import sys

epsilon = sys.float_info.epsilon

f = input('Ingrese Funcion a calcular\n')
x = input('Derivar esta funcion sobre: ')
xvalor = float(input('Valor de x: '))
deltax = float(input('Valor de deltax: '))

derivada = d.pDeriv(f, x)
der = derivada.subs(x, 2)

array = np.arange(0, 1, deltax, dtype=int)

array = np.linspace(0, 1, len(array), dtype=float)

for i in range(len(array)):
    limitexd = eval(f, array[i])
    fx_dx = (x, array[i] + deltax)
    fx = limitexd.subs(x, array[i])
    limite = (fx_dx - fx)/deltax
    print(f"Limite: {limite}")
    print(f"Derivada: {der}")
    if abs(der - limite) < epsilon: # 2.220446049250313e-16
        print("La derivada es correcta")

# lim deltax->0: f(x + deltax) - f(x)/deltax 

#x**3 + x + 1