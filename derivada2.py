#
# Codigo De Derivada Numerica para resolucion de problemas de Programacion No Lineal
# por Nicolas Barros D.
#

import derivada as d
from sympy import diff, Symbol
from sympy.parsing.sympy_parser import parse_expr

f = input('Ingrese Funcion a calcular\n')
x = input('Derivar esta funcion sobre: ')

derivada = d.pDeriv(f, x)


#x**3 + x + 1