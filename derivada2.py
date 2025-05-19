#
# Derivada Numérica con estimación de Δx óptimo
# Autor: Nicolás Barros D.
#

import numpy as np
import sympy as sp
import sys

epsilon = sys.float_info.epsilon
print(f"Valor de ε (error absoluto): {epsilon:.2e}\n")

# x**3 + x + 1
f_input = input("Ingrese la función f(x): ")

x_valor = float(input("Valor de x para evaluar la derivada: "))

x_precision = 0

x = sp.Symbol("x")
f = sp.sympify(f_input)

f_deriv = sp.diff(f, x)
deriv_real = f_deriv.evalf(subs={x: x_valor})

print(f"\nDerivada simbólica: f'({x_valor}) = {deriv_real}\n")

# Probar distintos Δx
print("Δx\t\tDerivada Numérica\tError Absoluto")

mejor_dx = None


dx = 1.0

while dx > epsilon:
    derivada_aprox = sp.limit((f.evalf(subs={x: x_valor + dx}) - f.evalf(subs={x: x_valor}))/dx, dx, 0)

    derivada_real = f_deriv.evalf(subs={x: x_valor})
    error = abs(derivada_aprox - derivada_real)
    print("derivada_aprox", derivada_aprox)
    print("derivada_real", derivada_real)
    print("error", error)

    if error < epsilon:
        print(f"dx óptimo encontrado: {dx}")
        break

    dx /= 2

print(f"dx óptimo: {dx}")
print(f"Derivada numérica: {derivada_aprox}")
print(f"Derivada analítica: {derivada_real}")
print(f"Error absoluto: {error}")