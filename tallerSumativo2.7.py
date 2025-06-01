import sympy as sp
import numpy as np

def analizar_puntocrtico(f_expr, punto):
    x, y = sp.symbols('x y')
    
    f_xx = sp.diff(f_expr, x, x)
    f_yy = sp.diff(f_expr, y, y)
    f_xy = sp.diff(f_expr, x, y)
    f_yx = sp.diff(f_expr, y, x)
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

# Ejemplo de uso
x, y = sp.symbols('x y')
f = x**2 + 2*y**2  # Mínimo en (0, 0)
punto = (0, 0)

resultado = analizar_puntocrtico(f, punto)
print("Resultado:", resultado)