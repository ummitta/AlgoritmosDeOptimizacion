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
        print("asdasdasdasdasd ", columnasLetras)

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
    filas, columnas = tablero.shape
    iteracion = 0
    print("Iniciando pivote para todos los casos")

    while True:
        iteracion += 1
        print(f"\n🔁 Iteración {iteracion}")
        filaZ = tablero[0, 1:-1]
        print("Fila Z (excluyendo Z y LD):", filaZ)

        # Identificar columnas con coeficientes negativos válidos
        negativos = [
            j + 1 for j, val in enumerate(filaZ)
            if val < -epsilon and columnasLetras[j + 1].startswith(('x', 'h', 'e'))
        ]

        if not negativos:
            print("✅ No quedan coeficientes negativos en Z. Solución óptima encontrada.")
            break

        # Seleccionar variable entrante: la más negativa
        j_ent = min(negativos, key=lambda j: tablero[0, j])
        print(f"➡️ Variable entrante: {columnasLetras[j_ent]} (columna {j_ent})")

        # Calcular ratios
        ratios = np.full(filas, np.inf)
        for i in range(1, filas):
            aij = tablero[i, j_ent]
            if aij > epsilon:  # solo positivos
                ratios[i] = tablero[i, -1] / aij

        

        i_sal = np.argmin(ratios[1:]) + 1
        print(f"⬅️ Variable saliente: {filasLetras[i_sal]} (fila {i_sal})")

        # Actualizar variable básica
        filasLetras[i_sal] = columnasLetras[j_ent]

        pivote = tablero[i_sal, j_ent]
        if abs(pivote) < epsilon:
            print("⚠️ Pivote cercano a cero. Se detiene para evitar inestabilidad.")
            break

        # Normalizar fila pivote
        tablero[i_sal, :] /= pivote

        # Eliminar el resto de la columna pivote
        for i in range(filas):
            if i != i_sal:
                factor = tablero[i, j_ent]
                tablero[i, :] -= factor * tablero[i_sal, :]

        tablero[np.abs(tablero) <= epsilon] = 0.0
        print(f"✅ Pivote aplicado en fila {i_sal}, columna {j_ent}")
        ImprimirTabla(tablero, filasLetras, columnasLetras)

    return tablero

def Pivoteo2(tablero, filasLetras, columnasLetras):
    print("Iniciando Pivoteo2...")

    for i in range(1, len(filasLetras)):  # Saltar Z
        var_basica = filasLetras[i]

        if var_basica not in columnasLetras:
            print(f"Variable {var_basica} no está en columnas, se ignora.")
            continue

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

def Pivoteo3(tablero, filasLetras, columnasLetras):
    filas, columnas = tablero.shape
    print("Iniciando pivoteo 3")

    while True:
        # Extraer coeficientes de la fila objetivo (Z), omitiendo columna 0 y la última
        z = tablero[0, 1:columnas-1]
        print("Fila Z (excluyendo Z y LD):", z)

        # Si no hay negativos, la solución es óptima
        if np.all(z >= 0):
            print("Solución óptima alcanzada")
            break

        # Columna entrante: índice del coeficiente más negativo en Z
        j_ent = np.argmin(z) + 1
        print(f"Columna entrante: {columnasLetras[j_ent]} (índice {j_ent})")

        # Cálculo de ratios para elegir fila saliente
        ratios = np.full(filas, np.inf)
        for i in range(1, filas):
            denom = tablero[i, j_ent]
            if denom > epsilon:
                ratios[i] = tablero[i, -1] / denom
        print("Ratios:", ratios)

        # Fila saliente: la de menor ratio positivo
        i_sal = np.argmin(ratios[1:]) + 1
        print(f"Fila saliente: {filasLetras[i_sal]} (índice {i_sal})")

        # Actualizar variable básica
        filasLetras[i_sal] = columnasLetras[j_ent]

        # Normalizar la fila pivote
        pivote = tablero[i_sal, j_ent]
        tablero[i_sal, :] /= pivote

        # Eliminar entradas en la columna pivote en otras filas
        for i in range(filas):
            if i != i_sal:
                factor = tablero[i, j_ent]
                tablero[i, :] -= factor * tablero[i_sal, :]

        # Mostrar tabla intermedia
        ImprimirTabla(tablero, filasLetras, columnasLetras)


    
    tablero[0, 1:] *= -1
    return tablero

def elimVarArtificiales(tablero, filasLetras, columnasLetras):
    print("Eliminar variables artificiales")

    # Identificar los índices de columnas artificiales
    indices_a_eliminar = [i for i, nombre in enumerate(columnasLetras) if nombre.startswith('a')]

    # Eliminar columnas del tablero y nombres correspondientes
    for idx in sorted(indices_a_eliminar, reverse=True):
        tablero = np.delete(tablero, idx, axis=1)
        columnasLetras.pop(idx)

    print("VARIABLES ARTIFICIALES ELIMINADAS")
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
    print("Simplex")
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
    print("Despues de forma ampliada")
    print(AA)
    aux = True
    while aux: #maximizacion menor igual
        zFuncion = AA[0, 1:-1]
        aux = hay_negativos_validos(zFuncion, columnasLetras, procedimiento, AA)
        if not aux and procedimiento == "2fases":
            print("SIMPLEX 2 FASES DETECTADO")
            AA = elimVarArtificiales(AA, filasLetras, columnasLetras)
            AA = reconstruirSimplex2(AA, c, obj, filasLetras, columnasLetras)
            print("despues de reconstruir en fase 2")
            ImprimirTabla(AA,filasLetras,columnasLetras)
            print("procedimiento: ", procedimiento)
            AA = Pivoteo2(AA,filasLetras,columnasLetras)
            print("Despues de pivoteo 2")
            ImprimirTabla(AA,filasLetras,columnasLetras)

            # SOLO SI HAY NEGATIVOS
            zFuncion = AA[0, 1:-1]
            aux = hay_negativos_validos(zFuncion, columnasLetras, procedimiento, AA)
            if not aux:
                AA = Pivoteo3(AA,filasLetras,columnasLetras)
                print("Despues de pivoteo 3")
                ImprimirTabla(AA,filasLetras,columnasLetras)
                # Vuelve a revisar por si acaso
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
                AA = Pivoteo3(AA,filasLetras,columnasLetras)
                print("Despues de pivoteo 3")
                ImprimirTabla(AA,filasLetras,columnasLetras)
                zFuncion = AA[0, 1:-1]
                aux = hay_negativos_validos(zFuncion, columnasLetras, procedimiento, AA)
            continue


        if not aux and procedimiento == "mixto" and obj == "max":
            print("cambiando a simplex mixto (max)")

            AA = elimVarArtificiales(AA, filasLetras, columnasLetras)
            procedimiento = "simplex"

            #print("AA :", AA)
            # Pasar c como está, sin modificarlo más
            #c = -1*c #si sale positivo no se borra, pero sino se borra
            AA = reconstruirSimplex(AA, c, "min", filasLetras, columnasLetras,procedimiento)
            print("antes de pivoteo2: ", AA)
            AA = Pivoteo2(AA,filasLetras,columnasLetras)

            print(AA)
            aux = True
            continue

        print("antes del if de aux :" , aux)
        ImprimirTabla(AA,filasLetras,columnasLetras)
        if aux:
            AA = Pivoteo(AA, filasLetras, columnasLetras)
    return AA

def ImprimirTabla(tablero, filasLetras, columnasLetras):
    print("Dentro de imprimir tabla")
    print("filasLetras: ", filasLetras)
    print("columnasLetras: ", columnasLetras)
    datos = [[fila] + list(fila_valores) for fila, fila_valores in zip(filasLetras, tablero)] 
    encabezados = [""]
    for i in range(len(columnasLetras)):
        encabezados.append(columnasLetras[i])
    print(tabulate(datos, headers=encabezados, tablefmt="grid"))

def parser(restriccion_str, v):
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