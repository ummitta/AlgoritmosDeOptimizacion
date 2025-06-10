# Taller Sumativo 2.7: Tipo de punto crítico
# Autores: Nicolas Barros, Maximo Mora

import sympy as sp
import numpy as np

def analizar_puntocrtico(f_expr, punto, deltaxs):
    x, y = sp.symbols('x y')

    f_expr = f.subs({x: x + deltaxs[0], y: y + deltaxs[1]})
    f_expr = sp.simplify(f_expr)

    # creacion de la función simbólica
    f_xx = sp.diff(f_expr, x, x)
    f_yy = sp.diff(f_expr, y, y)
    f_xy = sp.diff(f_expr, x, y)
    f_yx = sp.diff(f_expr, y, x)

    # creacion de la matriz Hessiana
    f_xx_val = float(f_xx.subs({x: punto[0], y: punto[1]}))
    f_yy_val = float(f_yy.subs({x: punto[0], y: punto[1]}))
    f_xy_val = float(f_xy.subs({x: punto[0], y: punto[1]}))
    f_yx_val = float(f_yx.subs({x: punto[0], y: punto[1]}))
    
    H = np.array([[f_xx_val, f_xy_val],
                  [f_yx_val, f_yy_val]])
    
    det_H = np.linalg.det(H)
    
    print("Hessiana en el punto:")
    print(H)
    print(f"Determinante: {det_H:.4f}")
    
    if det_H > 0:
        if f_xx_val > 0:
            return "Mínimo local"
        elif f_xx_val < 0:
            return "Máximo local"
        else:
            return "No hay información suficiente"
    elif det_H < 0:
        return "Punto de silla"
    else:
        return "No hay información suficiente"

x, y = sp.symbols('x y')
f_input = input("Ingrese la función f(x, y): ")
f = sp.sympify(f_input)
punto_critico = tuple(map(float, input("Ingrese el punto crítico (x, y): ").strip().split(',')))
deltaxs = tuple(map(float, input("Ingrese el valor de los deltax: ").strip().split(',')))

resultado = analizar_puntocrtico(f, punto_critico, deltaxs)
print("Resultado:", resultado)