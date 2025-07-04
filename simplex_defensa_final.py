# Algoritmo Simplex para resolver problemas de programación lineal
# Autores: Nicolas Barros, Maximo Mora

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
import sympy as sp

epsilon = sys.float_info.epsilon
np.set_printoptions(suppress=True, precision=2)

def hay_negativos_validos(filaZ, columnasLetras, fase, tablero):
    epsilon = 1e-7
    for i, valor in enumerate(filaZ):
        col_name = columnasLetras[i + 1]  # +1 por el desplazamiento por Z
        if valor < -epsilon:
            if fase == "2fases" and col_name.startswith('a'):
                return True
            if fase in ("simplex", "mixto") and (col_name.startswith('x') or col_name.startswith('h')):
                return True
    return False

def FormaAmpliada(A, b, c, ci, signos, obj, metodo, filasLetras, columnasLetras):
    # Forma Ampliada, toma A, b, c, signos y devuelve el tablero inicial
    zValor = 0
    if obj == "max":
        zValor = 1
    elif obj == "min":
        zValor = -1
        c = -1*c

    filasLetras.append("Z")
    columnasLetras.append("Z")
    holguras = 0
    excesos = 0
    for i in range(len(A[0])):
        columnasLetras.append('x' + str(i+1))

    for i in range(len(signos)):
        if signos[i] == 1:
            columnasLetras.append('h' + str(i+1))
            filasLetras.append('h' + str(i+1))
            holguras += 1
        if signos[i] == 3:
            columnasLetras.append('e' + str(i+1))
            columnasLetras.append('a' + str(i+1))
            filasLetras.append('e' + str(i+1))
            excesos += 2

    columnasLetras.append('LD')

    zFila = np.zeros((1, len(columnasLetras)), dtype=float)

    if metodo == "2fases":
        print("metodo 2 fases")

        wFila = np.zeros_like(zFila)
        wFila[0, 0] = zValor  # W en la posición [0][0]

        for i in range(len(signos)):
            if signos[i] == 3:
                filaRestriccion = np.zeros((1, len(columnasLetras)), dtype=float)
                filaRestriccion[0, 1:1+len(A[i])] = A[i]
                filaRestriccion[0, -1] = b[i]

                indiceE = columnasLetras.index('e' + str(i+1)) # Agregar exceso (-1) y artificial (+1)
                indiceA = columnasLetras.index('a' + str(i+1))

                filaRestriccion[0, indiceE] = -1
                filaRestriccion[0, indiceA] = 1

                wFila = wFila + filaRestriccion

        zFila = -1 * wFila
    if metodo == "simplex":
        zFila[0, 0] = zValor  # Z en la posición [0][0]
        for i in range(np.shape(c)[1]):
            zFila[0][i+1] = c.flatten()[i]
    if metodo == "mixto":
        print("metodo mixto activado")

        wFila = np.zeros_like(zFila)
        wFila[0, 0] = zValor

        for i in range(len(signos)):
            if signos[i] == 3:
                filaRestriccion = np.zeros((1, len(columnasLetras)), dtype=float)
                filaRestriccion[0, 1:1 + len(A[i])] = A[i]
                filaRestriccion[0, -1] = b[i]

                indiceE = columnasLetras.index('e' + str(i + 1))
                indiceA = columnasLetras.index('a' + str(i + 1))

                filaRestriccion[0, indiceE] = -1  # exceso con -1

                wFila = wFila + filaRestriccion
        # Crear Z = -W para usar como fila inicial en el tablero (fase 1 parcial)
        print("Zfila dentro de modo mixto: ", wFila)
        zFila = -1 * wFila
        print("Zfila dentro de modo mixto: ", zFila)
        
    filas = np.shape(b)[0] + 1
    columnas = len(columnasLetras)
    tablero = np.zeros((filas, columnas), dtype=float)

    tablero[0, :] = zFila
    print("np.shape(b)[0]: ", np.shape(b)[0])

    for i in range(np.shape(b)[0]):
        for j in range(np.shape(A)[1]):
            tablero[i+1][j+1] = A[i][j]
    
    tablero[1:filas, -1] = b.flatten()

    variables = np.zeros((np.shape(b)[0], np.shape(c)[1] + holguras + excesos))
    separacion = np.shape(c)[1]
    print("separacion: ", separacion)

    for i in range(np.shape(variables)[0]):
        for j in range(np.shape(b)[0]-1):
            if signos[i] == 1:
                variables[i][separacion] = 1
                tablero[i+1][separacion+1] = 1
                separacion += 1
                break

            if signos[i] == 3:
                variables[i][separacion] = -1
                variables[i][separacion+1] = 1
                tablero[i+1][separacion+1] = -1
                tablero[i+1][separacion+2] = 1
                separacion += 2
                break

    print("variables: \n", variables)
    print("tablero: \n", tablero)
    print("Imprimir tabla en forma ampliada, justo antes de retornar ")
    ImprimirTabla(tablero, filasLetras, columnasLetras)
    
    return tablero

def Pivoteo(tablero, filasLetras, columnasLetras):
    # Pivoteo para todos los casos, toma el tablero y las letras de filas y columnas
    filas, columnas = tablero.shape
    iteracion = 0
    print("Iniciando pivote para todos los casos")

    while True:
        print(f"Pivoteando...")
        filaZ = tablero[0, 1:-1]

        negativos = [
            j + 1 for j, val in enumerate(filaZ)
            if val < -epsilon and columnasLetras[j + 1].startswith(('x', 'h', 'e'))
        ]

        if not negativos:
            print("No quedan coeficientes negativos en Z. Solución óptima encontrada.")
            break

        j_ent = min(negativos, key=lambda j: tablero[0, j])

        ratios = np.full(filas, np.inf)
        for i in range(1, filas):
            aij = tablero[i, j_ent]
            if aij > epsilon:
                ratios[i] = tablero[i, -1] / aij

        i_sal = np.argmin(ratios[1:]) + 1

        print("Pivoteo aplicado para variable:", filasLetras[i_sal], "en columna:", columnasLetras[j_ent])

        filasLetras[i_sal] = columnasLetras[j_ent]

        pivote = tablero[i_sal, j_ent]
        if abs(pivote) < epsilon:
            print("Pivote cercano a cero. Se detiene para evitar inestabilidad.")
            break

        tablero[i_sal, :] /= pivote

        for i in range(filas):
            if i != i_sal:
                factor = tablero[i, j_ent]
                tablero[i, :] -= factor * tablero[i_sal, :]

        tablero[np.abs(tablero) <= epsilon] = 0.0
        print(f"Pivote aplicado en fila {i_sal}, columna {j_ent}")
        ImprimirTabla(tablero, filasLetras, columnasLetras)

    return tablero

def Pivoteo2(tablero, filasLetras, columnasLetras):
    # Pivoteo para la segunda fase del metodo de 2 fases, toma el tablero y las letras de filas y columnas

    for i in range(1, len(filasLetras)):  # Saltar Z
        var_basica = filasLetras[i]

        if var_basica not in columnasLetras:
            print(f"Variable {var_basica} no está en columnas, se ignora.")
            continue

        print("Pivoteando...")

        fila_idx = i
        col_idx = columnasLetras.index(var_basica)

        pivote = tablero[fila_idx][col_idx]
        if abs(pivote) < 1e-8:
            print(f"La posición de pivote ({var_basica}, {var_basica}) es cero, no se puede normalizar.")
            continue

        tablero[fila_idx] = tablero[fila_idx] / pivote

        for j in range(len(tablero)):
            if j != fila_idx:
                factor = tablero[j][col_idx]
                tablero[j] = tablero[j] - factor * tablero[fila_idx]

        print(f"Pivoteo aplicado para variable básica '{var_basica}' en columna '{columnasLetras[col_idx]}'.")

    return tablero

def elimVarArtificiales(tablero, filasLetras, columnasLetras):
    indices_a_eliminar = [i for i, nombre in enumerate(columnasLetras) if nombre.startswith('a')]
    for idx in sorted(indices_a_eliminar, reverse=True):
        tablero = np.delete(tablero, idx, axis=1)
        columnasLetras.pop(idx)
    ImprimirTabla(tablero, filasLetras, columnasLetras)
    return tablero

def reconstruirSimplex(tablero, c, obj, filasLetras, columnasLetras, procedimiento):
    zValor = 0
    if obj == "max":
        zValor = 1
    elif obj == "min":
        zValor = -1
        c = -1*c

    zFila = np.zeros((1, len(columnasLetras)), dtype=float)

    for i in range(np.shape(c)[1]):
            zFila[0][i+1] = c.flatten()[i]

    #agregar zfila al tablero
    print("zfila: ", zFila)
    tablero[0, 0] = zValor
    if procedimiento == "2fases":
        tablero[0, :] = -1* zFila
    else:
        tablero[0, :] = zFila
    #tablero[0, -1] = 0
    print("tablero antes de agregar la fila Z: ", tablero)

    return tablero

def reconstruirSimplex2(tablero, c, obj, filasLetras, columnasLetras):
    if obj == "max":
        zValor = 1
        
    elif obj == "min":
        zValor = -1
        c = -1*c

    zFila = np.zeros((1, len(columnasLetras)), dtype=float)

    for i in range(np.shape(c)[1]):
            zFila[0][i+1] = c.flatten()[i]

    #agregar zfila al tablero
    tablero[0, :] = -1*zFila
    tablero[0, 0] = zValor
    #tablero[0, -1] = 0


    return tablero

def Simplex(A, b, c, ci, signos, obj, filasLetras, columnasLetras):
    procedimiento = ''
    aux = False
    menorIgual = 0
    mayorIgual = 0

    for i in range(len(signos)):
        if signos[i] == 1:
            menorIgual += 1
        elif signos[i] == 3:
            mayorIgual += 1

    if menorIgual > 0 and mayorIgual == 0:
        procedimiento = 'simplex'
        c = -1*c

    if menorIgual == 0 and mayorIgual > 0:
        procedimiento = '2fases'

    if menorIgual > 0 and mayorIgual > 0:
        procedimiento = "mixto"

    AA = FormaAmpliada(A, b, c, ci, signos, obj, procedimiento, filasLetras, columnasLetras)

    aux = True
    while aux: # Maximizacion menor igual
        zFuncion = AA[0, 1:-1]
        aux = hay_negativos_validos(zFuncion, columnasLetras, procedimiento, AA)
        if not aux and procedimiento == "2fases":
            print("SIMPLEX 2 FASES DETECTADO")
            AA = elimVarArtificiales(AA, filasLetras, columnasLetras)
            AA = reconstruirSimplex2(AA, c, obj, filasLetras, columnasLetras)
            ImprimirTabla(AA,filasLetras,columnasLetras)
            AA = Pivoteo2(AA,filasLetras,columnasLetras)
            ImprimirTabla(AA,filasLetras,columnasLetras)
            zFuncion = AA[0, 1:-1]
            aux = hay_negativos_validos(zFuncion, columnasLetras, procedimiento, AA)
            continue

        if not aux and procedimiento == "mixto" and obj == "min":
            AA = elimVarArtificiales(AA, filasLetras, columnasLetras)
            procedimiento = "2fases"
            AA = reconstruirSimplex(AA, c, "min", filasLetras, columnasLetras,procedimiento)
            AA = Pivoteo2(AA,filasLetras,columnasLetras)
            ImprimirTabla(AA,filasLetras,columnasLetras)
            zFuncion = AA[0, 1:-1]
            aux = hay_negativos_validos(zFuncion, columnasLetras, procedimiento, AA)
            
            if not aux:
                AA = Pivoteo(AA,filasLetras,columnasLetras)
                print("Despues de pivoteo 3")
                ImprimirTabla(AA,filasLetras,columnasLetras)
                zFuncion = AA[0, 1:-1]
                aux = hay_negativos_validos(zFuncion, columnasLetras, procedimiento, AA)
            continue

        if not aux and procedimiento == "mixto" and obj == "max":
            AA = elimVarArtificiales(AA, filasLetras, columnasLetras)
            procedimiento = "simplex"
            AA = reconstruirSimplex(AA, c, "min", filasLetras, columnasLetras,procedimiento)
            AA = Pivoteo2(AA,filasLetras,columnasLetras)
            aux = True
            continue

        ImprimirTabla(AA,filasLetras,columnasLetras)
        if aux:
            AA = Pivoteo(AA, filasLetras, columnasLetras)
    return AA

def ImprimirTabla(tablero, filasLetras, columnasLetras):
    # Imprime el tablero en formato tabla
    datos = [[fila] + list(fila_valores) for fila, fila_valores in zip(filasLetras, tablero)] 
    encabezados = [""]
    for i in range(len(columnasLetras)):
        encabezados.append(columnasLetras[i])
    print(tabulate(datos, headers=encabezados, tablefmt="grid"))

def parser(restriccion_str, v):
    # Analiza una restriccion en formato de cadena y devuelve coeficientes, signo y lado derecho
    variables = sp.symbols(' '.join([f'x{i+1}' for i in range(v)]))
    expresion = sp.sympify(restriccion_str)
    ladoIzquierdo, ladoDerecho = expresion.lhs, expresion.rhs
    if expresion.has(sp.LessThan):
        signo = 1
    elif expresion.has(sp.GreaterThan):
        signo = 3
    else:
        signo = 5
    coeficientes = [float(expresion.lhs.coeff(var)) for var in variables]
    lado_derecho = float(expresion.rhs)
    return coeficientes, signo, lado_derecho

def parse_obj(obj_str, v):
    variables = sp.symbols(' '.join([f'x{i+1}' for i in range(v)]))
    expresion = sp.sympify(obj_str)
    coeficientes = [float(expresion.coeff(var)) for var in variables]
    return coeficientes

def menu():
    columnasLetras = []
    filasLetras = []
    v = int(input("Cantidad de variables: "))
    if v <= 0:
        print("Las variables deben ser > 0")
        return
    n = int(input("Cantidad de restricciones: "))
    if n <= 0:
        print("Las restricciones deben ser > 0")
        return
    A = np.empty((0, v))
    b = np.empty((0, 1))
    signos = np.empty((0,1))
    ci = np.empty((0,1))
    print("Ingresa cada restricción, formato:\n2*x1 + 3*x2 <= 300\n")
    for i in range(n):
        línea = input(f"> ")
        coefs, signo, rhs = parser(línea, v)
        print(coefs, signo, rhs)

        A = np.vstack([A, coefs])
        print("A: ", A)
        b = np.vstack([b, [rhs]])
        print("b: ", b)
        print("signos: ", signos)
        print("signo: ", signo)
        signos = np.vstack([signos, [signo]])
        print("signos: ", signos)
    print("Ahora ingresa la función obj, formato:\n30000*x1 + 4000*x2\n")
    obj_str = input("> ")
    c = np.array(parse_obj(obj_str, v)).reshape(1, -1)
    print("Ahora el tipo de modo 'max' o 'min'\n")
    modo = input(">")
    print("Matriz A (coeficientes de restricciones):\n")
    print(A)
    print("Vector b (lado derecho de restricciones):\n")
    print(b)
    print("Signos de las restricciones:")
    print(signos)
    print("Vector c (coeficientes de la función obj):\n")
    print(c)
    return Simplex(A, b, c, ci, signos, modo, filasLetras, columnasLetras)

tablero = menu()