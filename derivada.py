#
# Codigo De Derivada Numerica para resolucion de problemas de Programacion No Lineal
# por Nicolas Barros D.
#

from tokenize import String
from sympy import diff, Symbol, evaluate, lambdify
from sympy.parsing.sympy_parser import parse_expr
from sympy import sympify
from sympy import limit

def calcDeriv(funct, x):
    xS = Symbol('x')
    aa = diff(funct, xS, evaluate=True)
    f = lambdify(xS, aa)
    return aa, f(xS)

def pDeriv(fIn, x):
    pDeriv = calcDeriv(fIn, x)

    return pDeriv

def limitex(fIn, x):
    xS = Symbol('x')
    limitexd = limit(fIn, xS, 0)
    return limitexd