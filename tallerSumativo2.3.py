# Taller Sumativo 2.3: Derivada numerica
# Autores: Nicolas Barros, Maximo Mora

import numpy as np
import sympy as sp
import sys

epsilon = sys.float_info.epsilon

print(f"Valor de epsilon: {epsilon:.2e}\n")

# ej: x**3 + x + 1
f_input = input("Ingrese la función f(x): ")

x_valor = float(input("Valor de x para evaluar la derivada: "))

x = sp.Symbol("x")
f = sp.sympify(f_input)

f_deriv = sp.diff(f, x)

derivada_real = f_deriv.evalf(subs={x: x_valor})

print(f"\nf'(x) = {f_deriv}\n")
print(f"f'({x_valor}) = {derivada_real}\n")

print("Δx\t\tDerivada Numérica\tDerivada Aproximada\tError Absoluto")
print("--------------------------------------------------------------------------------")


dx = 1.0
while dx > epsilon:
    derivada_aprox = (f.evalf(subs={x: x_valor + dx}) - f.evalf(subs={x: x_valor}))/dx # limite

    error = abs(derivada_aprox - derivada_real)

    print(f"{dx:.2e}\t{derivada_real}\t{derivada_aprox}\t{error}")

    if error < epsilon:
        print(f"dx óptimo encontrado: {dx}")
        break

    dx /= 2

print("Reporte de resultados:")
print("--------------------------")
print(f"dx óptimo: {dx}")
print(f"Derivada numérica: {derivada_aprox}")
print(f"Derivada analítica: {derivada_real}")
print(f"Error absoluto: {error}")