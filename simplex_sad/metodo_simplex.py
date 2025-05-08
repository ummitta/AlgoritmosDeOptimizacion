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
    elif obj == "min":
        zValor = -1

    # crear variables basicas
    filasLetras.append("Z")
    columnasLetras.append("Z")
    holguras = 0
    excesos = 0
    for i in range(np.shape(c)[0]):
        columnasLetras.append('x' + str(i+1))

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
    
    for i in range(len(c)):
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

    variables = np.zeros((np.shape(b)[0], np.shape(c)[0] + holguras + excesos))
    separacion = np.shape(c)[0]

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


    print("BuscarValoresNegativos ")
    valoresNegativos = False

    cocientesFuncion = 1
    for letra in filasLetras:

        if letra.startswith('x'):
            cocientesFuncion += 1

    zfuncion = np.zeros((1, len(tablero[0])), dtype=float)

    for i in range(1,len(tablero[0])):
        zfuncion[0][i] = tablero[0][i]


    for i in range(cocientesFuncion-1,np.shape(zfuncion)[1]-1):


        if zfuncion[0][i] < 0:
            valoresNegativos = True
 

    return  valoresNegativos

def Encontrar_col_pivote(tablero, filasLetras, columnasLetras):
    print("Encontrar_col_pivote")

    filaZ = tablero[0, 1:-1]
    valorMinimo = np.min(filaZ)
    indiceMinimo = np.argmin(filaZ) + 1
    #print(valorMinimo,indiceMinimo)

    filas = tablero.shape[0]
    operatoria = np.zeros((filas,1),dtype=float)
    columna = tablero.shape[1]

    for i in range(1,filas):
        numerador = tablero[i][columna-1]
        denominador = tablero[i][indiceMinimo]
        operacion = numerador / denominador
        if operacion > 0:
            print(f'{numerador} / {denominador} = {numerador / denominador}')

        
        if denominador != 0.0:
            operacion = numerador / denominador
            if operacion > 0:
                operatoria[i][0] = operacion 
        else:
            operatoria[i][0] = 666
            
        
    tablero = np.concatenate((tablero,operatoria),axis=1)
    
    ImprimirTabla(tablero, filasLetras, columnasLetras)
    print("filasLetras: ", filasLetras)
    print("columnasLetras: ", columnasLetras)

    return tablero

def Encontrar_fila_pivote(tablero, filasLetras, columnasLetras):

    print('Encontrar_fila_pivote')
    ImprimirTabla(columnasLetras,filasLetras,tablero)

    filaZ = tablero[0][:-2]
    valorMinimoZ = np.min(filaZ)
    print(valorMinimoZ)
    indiceMinimoZ = np.argmin(filaZ)
    filas = tablero.shape[0]
    columnas = tablero.shape[1]

    columnaOperacion = tablero[:, -1]

    print('columna Operacion 1 ', columnaOperacion)
    valorExcluir = 666
    columnaOperacionFiltrada = columnaOperacion[columnaOperacion != valorExcluir]
    columnaOperacionFiltrada = np.delete(columnaOperacionFiltrada,0,axis=0)
    columnaOperacionFiltrada = columnaOperacionFiltrada[columnaOperacionFiltrada > 0]
    valorMinimo = np.min(columnaOperacionFiltrada)

    indiceMinimo = np.where(columnaOperacion == valorMinimo)[0][0]

    print(f'indiceMinimoZ: {indiceMinimoZ}')
    print(f'indiceMinimo: {indiceMinimo}')
    pivote = tablero[indiceMinimo][indiceMinimoZ]

    print(pivote)
    print('letra fila: ', filasLetras[indiceMinimoZ])
    print('letra columna: ', columnasLetras[indiceMinimo])


    columnasLetras[indiceMinimo] = filasLetras[indiceMinimoZ]
    for i in range(columnas-1):

        tablero[indiceMinimo][i] = tablero[indiceMinimo][i] / pivote 

    ImprimirTabla(tablero, filasLetras, columnasLetras)
    return tablero 

def Pivotear(tablero, filasLetras, columnasLetras):
    print('Pivotear')
    
    filaZ = tablero[0][:-2]
    #print(f'fila z : {filaZ}')
    valorMinimoZ = np.min(filaZ)
    indiceMinimoZ = np.argmin(filaZ)
    filas = tablero.shape[0]
    columnas = tablero.shape[1]

    columnaOperacion = tablero[:, -1]
    valorExcluir = 666
    print('columnaOperacion ', columnaOperacion)

    columnaOperacionLimpia = columnaOperacion[(columnaOperacion > 0) & (columnaOperacion != 666)]
    print(columnaOperacion)
    print(columnaOperacionLimpia)

    valorMinimo = np.min(columnaOperacionLimpia)

    
    indiceMinimo = np.where(columnaOperacion == valorMinimo)[0][0]

    print('valorMinimo: ', valorMinimo)
    print('indiceMinimo: ', indiceMinimo)

    print(f"coordendas: {indiceMinimo} ,{indiceMinimoZ} ")

    ImprimirTabla(tablero, filasLetras, columnasLetras)
    
    for i in range(filas):
        pivoteColumna = tablero[i][indiceMinimoZ]
        for j in range(columnas-1):

        
            if i == indiceMinimo:
                break

            numero = tablero[i][j]
            pivote = tablero[indiceMinimo][j]

            operacion = numero + (pivoteColumna * -1* pivote)
            #print(f'operacion: {numero} - ({pivoteColumna} * {pivote} ) = {operacion}')
            #Casi cero
            if abs(operacion) <= epsilon:
                tablero[i][j] = 0
            else:
                tablero[i][j] = operacion
    tablero = np.delete(tablero, -1, axis=1)
    
    ImprimirTabla(tablero, filasLetras, columnasLetras)

    return tablero

def Simplex(A, b, c, ci, signos, obj, filasLetras, columnasLetras):
    print("Simplex")
    AA = FormaAmpliada(A, b, c, ci, signos, obj, filasLetras, columnasLetras)
    print("filasLetras: ", filasLetras)
    print("columnasLetras: ", columnasLetras)
    #Fase 1: solo si hay  artificiales y de exceso
    aux = False
    procedimiento = ''
    menorIgual = 0
    mayorIgual = 0
    for i in range(len(signos)):
        if signos[i] == 1:
            menorIgual += 1
        elif signos[i] == 3:
            mayorIgual += 1

    if menorIgual > 0 and mayorIgual == 0:
        procedimiento = 'simplex normal'
    if menorIgual == 0 and mayorIgual > 0:
        procedimiento = 'metodo de dos fases normal'
    if menorIgual > 0 and mayorIgual > 0:
        procedimiento = "metodo de fases mixto"
    

    if procedimiento == "simplex normal":
        aux = True

    while aux:
        AA = Encontrar_col_pivote(AA, filasLetras, columnasLetras)
        AA = Encontrar_fila_pivote(AA, filasLetras, columnasLetras)
        AA = Pivotear(AA, filasLetras, columnasLetras)
        
        zFuncion = AA[0, 1:-1]  
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

def parser(restriccion_str):
    x1, x2 = sp.symbols('x1 x2')
    expresion = sp.sympify(restriccion_str)
    # Separo LHS y RHS
    ladoIzquierdo, ladoDerecho = expresion.lhs, expresion.rhs

    if expresion.has(sp.LessThan):
        signo = 1
    elif expresion.has(sp.GreaterThan):
        signo = 3
    else:
        signo = 5

    coeficiente1 = ladoIzquierdo.coeff(x1, 1)
    coeficiente2 = ladoIzquierdo.coeff(x2, 1)
    
    return [float(coeficiente1), float(coeficiente2)], signo, float(ladoDerecho)

def parse_obj(obj_str):
    x1, x2 = sp.symbols('x1 x2')
    expresion = sp.sympify(obj_str)
    coeficiente1 = expresion.coeff(x1, 1)
    coeficiente2 = expresion.coeff(x2, 1)
    return [float(coeficiente1), float(coeficiente2)]

def menu():

    A = np.empty((0, 2))
    b = np.empty((0, 1))
    signos = np.empty((0,1))
    ci = np.empty((0,1))
     
    n = int(input("Cantidad de restricciones: "))
    if n <= 0:
        print("Las restricciones deben ser > 0")
        return
    
    print("Ingresa cada restricción, por ej: 2*x1 + 3*x2 <= 300")
    for i in range(n):
        línea = input(f"> ")
        coefs, signo, rhs = parser(línea)
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
    c = np.array(parse_obj(obj_str)).reshape(1, -1)

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

columnasLetras = []
filasLetras = []

A = np.array([[1,0],
              [0,2],
              [3,2],])

b = np.array([[4],
              [12],
              [18]])

#Porque es negativo?
c = np.array([30000, 50000])
#signos
# < 0
# <= 1
# > 2
# >= 3
# != 4
sign = np.array([[1],[1],[1]])

ci = np.array([[0]])

simplexResultado = Simplex(A, b, c, ci, sign, 'max', filasLetras, columnasLetras)
print("Resultado final: \n", simplexResultado[0][-1])