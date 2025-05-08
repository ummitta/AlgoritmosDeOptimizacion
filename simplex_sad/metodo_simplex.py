import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
import sympy as sp

epsilon = sys.float_info.epsilon
np.set_printoptions(suppress=True, precision=2)

def FormaAmpliada(A, b, c, ci, signos, obj, filasLetras, columnasLetras):
    zValor = 0
    if obj == "max":
        zValor = 1
        c = -1*c
        print("cc: ", c)
    elif obj == "min":
        zValor = -1

    # crear variables basicas
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
            columnasLetras.append('a' + str(i+1))
            columnasLetras.append('e' + str(i+1))
            filasLetras.append('e' + str(i+1))
            excesos += 2

    
    columnasLetras.append('LD')

    #Creacion de columna z (del largo de columnaletras)        
    zFila = np.zeros((1, len(columnasLetras)-1), dtype=float)
    zFila = np.insert(zFila, 0, zValor, axis=1)
    
    for i in range(np.shape(c)[1]):
        zFila[0][i+1] = c.flatten()[i]
    

    print("filasLetras ", filasLetras)
    print("columnasLetras ", columnasLetras)
    print("zFila ", zFila)

    #crear matriz vacia

    filas = np.shape(b)[0] + 1
    columnas = len(columnasLetras)
    tablero = np.zeros((filas, columnas), dtype=float)

    tablero[0, :] = zFila
    print("np.shape(b)[0]: ", np.shape(b)[0])
    print("A[3][1] ", A[2][1])

    for i in range(np.shape(b)[0]):
        for j in range(np.shape(A)[1]):
            tablero[i+1][j+1] = A[i][j]
    

    print("filas: ", filas)
    print("columnas: ", columnas)
    
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
                tablero[i+1][separacion+1] = -1
                separacion += 2
                break 

    print("variables: \n", variables)


    print("tablero: \n", tablero)

    ImprimirTabla(tablero, filasLetras, columnasLetras)
    
    return tablero

def Pivoteo(tablero, filasLetras, columnasLetras):
    print("Encontrar_col_pivote")
    valorExcluir = 666

    filas = tablero.shape[0]
    columnas = tablero.shape[1]
    filaZ = tablero[0, 1:-1]  # Excluye la primera y última columna

    # Crear un vector para almacenar las operaciones
    operatoria = np.zeros((filas, 1), dtype=float)

    # Buscar el índice y valor mínimo de la fila Z
    indiceMinimoZ = np.argmin(filaZ) + 1
    valorMinimoZ = np.min(filaZ)

    print(f'indiceMinimo: {indiceMinimoZ}')
    print(f'valorMinimoZ: {valorMinimoZ}')

    # Calcular las relaciones de los coeficientes de la restricción con la columna pivote
    columnaOperacion = tablero[:, -1]
    for i in range(1, filas):
        numerador = tablero[i][columnas-1]
        denominador = tablero[i][indiceMinimoZ]
        
        if denominador != 0.0:
            operacion = numerador / denominador
            if operacion > 0:
                operatoria[i][0] = operacion

            # Evitar divisiones por cero o valores numéricos muy pequeños
            if abs(denominador) <= epsilon:
                operatoria[i][0] = 0.0  # Asignar infinito para evitar división por cero

    # Imprimir tabla antes de pivotear
    ImprimirTabla(tablero, filasLetras, columnasLetras)

    
    print(f'indiceMinimoZ: {indiceMinimoZ}')
    #filtrar los valores que sean 0
    for i in range(1, filas):
        if operatoria[i][0] == 0.0:
            operatoria[i][0] = np.inf  

    indiceMinimo = np.argmin(operatoria[1:, 0]) + 1  # +1 para ajustar el índice
    # Realizar el pivote
    pivote = tablero[indiceMinimo][indiceMinimoZ]

    print(f'pivotexxxx: {pivote}')
    print(f'indicessss: {indiceMinimo, indiceMinimoZ}')

    filasLetras[indiceMinimo] = columnasLetras[indiceMinimoZ]

    for i in range(1, columnas):
        tablero[indiceMinimo][i] = tablero[indiceMinimo][i] / pivote


    ImprimirTabla(tablero, filasLetras, columnasLetras)

    print('Pivotear')

    # hacer ceros arriba y abajo del pivote
    for i in range(filas):
        pivoteColumna = tablero[i][indiceMinimoZ]
        for j in range(columnas):
            if i == indiceMinimo:
                break

            numero = tablero[i][j]
            pivote = tablero[indiceMinimo][j]
            operacion = numero + (pivoteColumna * -1 * pivote)

            # Evitar casi cero
            if abs(operacion) <= epsilon:
                tablero[i][j] = 0
            else:
                tablero[i][j] = operacion

    # Imprimir la tabla final
    ImprimirTabla(tablero, filasLetras, columnasLetras)

    return tablero

def Simplex(A, b, c, ci, signos, obj, filasLetras, columnasLetras):
    print("Simplex")

    procedimiento = ''
    aux = False
    AA = FormaAmpliada(A, b, c, ci, signos, obj, filasLetras, columnasLetras)
    
    #Fase 1: solo si hay  artificiales y de exceso
    
    
    menorIgual = 0
    mayorIgual = 0
    for i in range(len(signos)):
        if signos[i] == 1:
            menorIgual += 1
        elif signos[i] == 3:
            mayorIgual += 1

    if menorIgual > 0 and mayorIgual == 0:
        procedimiento = 'simplex'
    if menorIgual == 0 and mayorIgual > 0:
        procedimiento = 'metodo de dos fases normal'
    if menorIgual > 0 and mayorIgual > 0:
        procedimiento = "metodo de fases mixto"
    

    if procedimiento == "simplex":
        aux = True

    while aux:

        AA = Pivoteo(AA, filasLetras, columnasLetras)
        
        zFuncion = AA[0, 1:-1]
        print("zFuncion: ", zFuncion)
        aux = np.any(zFuncion < 0) 

    return AA

def ImprimirTabla(tablero, filasLetras, columnasLetras):
    # Crear tabla con encabezados y filas
    print("filasLetras: ", filasLetras)
    print("columnasLetras: ", columnasLetras)
    print("tablero: ", tablero)
    datos = [[fila] + list(fila_valores) for fila, fila_valores in zip(filasLetras, tablero)] 

    # Preparar encabezados: espacio vacío para esquina superior izquierda + columnas
    encabezados = [""]

    #imprimir las columnas
    for i in range(len(columnasLetras)):
        encabezados.append(columnasLetras[i])

    # Imprimir tabla formateada
    print(tabulate(datos, headers=encabezados, tablefmt="grid"))

def parser(restriccion_str, v):
    variables = sp.symbols(' '.join([f'x{i+1}' for i in range(v)]))
    # Convertir string a expresión
    expresion = sp.sympify(restriccion_str)
    # Separo LHS y RHS
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

    print("Ingresa cada restricción, por ej: 2*x1 + 3*x2 <= 300")
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

    # 2) Leer función obj
    print("\nAhora ingresa la función obj, ej: 30000*x1 + 4000*x2")
    obj_str = input("> ")
    c = np.array(parse_obj(obj_str, v)).reshape(1, -1)

    print("\nAhora el tipo de modo 'maximizar' o 'minimizar' ")

    modo = input(">")

    print("\nMatriz A (coeficientes de restricciones):")
    print(A)
    print("\nVector b (lado derecho de restricciones):")
    print(b)
    print("\nSignos de las restricciones:")
    print(signos)
    print("\nVector c (coeficientes de la función obj):")
    print(c)
    return Simplex(A, b, c, ci, signos, modo, filasLetras, columnasLetras)

columnasLetras = []
filasLetras = []

menu()

