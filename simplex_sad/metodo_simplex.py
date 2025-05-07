import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
import sympy as sp

epsilon = sys.float_info.epsilon
np.set_printoptions(suppress=True, precision=2)

columnasLetras = ['Z','x1','x2']
filasLetras = ['Z']

def FormaAmpliada(A, b, c, ci, signos, objetivo, filasLetras, columnasLetras):
    #Creacion de columna Z
    zValor = 0
    if objetivo == "maximizar":
        zValor = 1

        c = -1*c
    elif objetivo == "minimizar":
        zValor = -1
        
    #Creacion de columna z para ver si es maximizacion o minizacion
    
    variablesBasicas = np.zeros((b.shape[0],1),dtype=float)
    zColumna = np.vstack((zValor, variablesBasicas))
    
    #Creacion de matriz de coeficiente, y los posibles signos

    signosLimite = signos[:A.shape[0]]

    cantidadHolguras = np.count_nonzero(signosLimite == 1)
    cantidadArtificiales = np.count_nonzero(signosLimite == 3)

    cantidadExceso = cantidadArtificiales * 2

    cantidadVariablesTotales = cantidadHolguras + cantidadExceso

    cantidadCoeficientes = np.shape(A)[0]-1
    
    variables = np.zeros((np.shape(b)[0], cantidadCoeficientes + cantidadVariablesTotales))

    separacion = cantidadCoeficientes

    for i in range(np.shape(variables)[0]):
        for j in range(np.shape(b)[0]-1):
            for k in range(np.shape(A)[1]):
                variables[i][k] = A[i][k]
            

            if signos[i] == 1:
                variables[i][separacion] = 1
                separacion += 1

                break

            if signos[i] == 3:
                variables[i][separacion] = -1
                variables[i][separacion+1] = 1
                separacion += 2

                break 

    # Creación columna LD
    ldColumna = np.concatenate((ci, b), axis=0)

    # FIX: usar columnas (shape[1]) no filas (shape[0])
    zFila = np.zeros((1, variables.shape[1]))  
    zFila[0, :c.shape[0]] = c.flatten()  # Copiar coeficientes de c

     # 1) Creamos la fila Z con los coeficientes de c:
    zFila = np.zeros((1, variables.shape[1]), dtype=float)
    zFila[0, :c.size] = c.flatten()  # ahora zFila.shape == (1, n)

    # 2) Apilamos zFila sobre variables para tener (m+1) filas:
    zYvars = np.vstack((zFila, variables))  
    # zYvars.shape == (m+1, n)

    # 3) Concatenamos la columna Z a la izquierda:
    #    zColumna.shape == (m+1, 1)
    Z_and_Y = np.hstack((zColumna, zYvars))
    # Z_and_Y.shape == (m+1, 1 + n)

    # 4) Preparamos la columna LD:
    #    ci.shape == (1,)   b.shape == (m,)
    ld_array = np.concatenate((ci.flatten(), b.flatten()))
    columnaZvariables = ld_array.reshape(-1, 1)   
    # ldColumna.shape == (m+1, 1)

    # Ajustar ldColumna si filas no coinciden
    if columnaZvariables.shape[0] > ldColumna.shape[0]:
        extra = columnaZvariables.shape[0] - ldColumna.shape[0]
        ldColumna = np.vstack([ldColumna,
                               np.zeros((extra, ldColumna.shape[1]))])

    tablero = np.concatenate((columnaZvariables, ldColumna), axis=1)

    # Generar etiquetas de columnas (holguras, excesos, artificiales…)
    if cantidadHolguras > 0 and cantidadArtificiales == 0:
        for i in range(cantidadHolguras):
            columnasLetras.append(f'h{i+1}')

    if cantidadArtificiales > 0:
        # En ambos casos (solo artificiales o ambos tipos)
        for i in range(cantidadArtificiales):
            columnasLetras += [f'e{i+1}', f'a{i+1}']
        if cantidadHolguras > 0:
            for i in range(cantidadHolguras):
                columnasLetras.append(f'h{i+1}')

    columnasLetras.append('LD')

    # FIX: pasarle filasLetras y columnasLetras
    ImprimirTabla(tablero, filasLetras, columnasLetras)

    return tablero

def BuscarValoresNegativos(tablero):
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

def Encontrar_col_pivote(tablero,posicionColumna=0):
    global filas
    global columnas
    print("Encontrar_col_pivote")
    print(tablero)

    print(tablero[0, 1:-1])

    filaZ = tablero[0, 1:-1]
    valorMinimo = np.min(filaZ)
    indiceMinimo = np.argmin(filaZ) + 1

    if posicionColumna == 0:
        print("posicion columna es igual a cero")
        filaZ = tablero[0, 1:-1]
        valorMinimo = np.min(filaZ)
        indiceMinimo = np.argmin(filaZ) + 1
        print('valor minimo',valorMinimo)

        

    if posicionColumna != 0:
        print("posicion columna distinta de cero")
        indiceMinimo = posicionColumna
    

    filas = tablero.shape[0]
    operatoria = np.zeros((filas,1),dtype=float)
    columna = tablero.shape[1]
    indiceMinimo = int(indiceMinimo)
    
    print("filaZ: ", filaZ)
    print("tablero[0][indiceMinimo]: ",filaZ[int(indiceMinimo)])
    print("indiceMinimo: ",indiceMinimo)
    for i in range(1,filas):
        numerador = tablero[i][columna-1]
        denominador = tablero[i][indiceMinimo]
        if posicionColumna != 0:
            denominador = tablero[i][indiceMinimo+1]
        operacion = numerador / denominador
        print(f"operacion = {numerador} / {denominador} = {operacion}")
        #no borrar
        if operacion > 0:
             print(f'{numerador} / {denominador} = {numerador / denominador}')

        
        if denominador != 0.0:
            operacion = numerador / denominador
            if operacion > 0:
                operatoria[i][0] = operacion 
        else:
            operatoria[i][0] = -1

    print(operatoria)    
    indexPositivo = np.where(operatoria > 0)[0]

    valorPositivos = operatoria[indexPositivo,0]
    valorMinimoMayor = valorPositivos[np.argmin(valorPositivos)]
    indiceFila = np.where(operatoria == valorMinimoMayor)[0]
    indiceFila = indiceFila[0]
    print("test indiceFila ", indiceFila)
    
    tablero = np.concatenate((tablero,operatoria),axis=1)
    
    filasLetras.append('Operacion')
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    filasLetras.pop()
    return tablero,indiceFila

def Encontrar_fila_pivote(tablero,filaPivote=0):

    print('Encontrar_fila_pivote')
    ImprimirTabla(columnasLetras,filasLetras,tablero)

    filaZ = tablero[0][:-2]
    valorMinimoZ = np.min(filaZ)
    print(valorMinimoZ)
    indiceMinimoZ = np.argmin(filaZ)

    indiceMinimo = 0

    filas = tablero.shape[0]
    columnas = tablero.shape[1]

    if filaPivote == 0:
            


            columnaOperacion = tablero[:, -1]

            print('columna Operacion 1 ', columnaOperacion)
            valorExcluir = -1
            columnaOperacionFiltrada = columnaOperacion[columnaOperacion != valorExcluir]
            columnaOperacionFiltrada = np.delete(columnaOperacionFiltrada,0,axis=0)
            columnaOperacionFiltrada = columnaOperacionFiltrada[columnaOperacionFiltrada > 0]
            valorMinimo = np.min(columnaOperacionFiltrada)

            indiceMinimo = np.where(columnaOperacion == valorMinimo)[0][0]


    if filaPivote != 0:
        print("filaPivote != 0")
        indiceMinimo = filaPivote



    print("indiceMinimo: ", indiceMinimo)
    # print(f'indiceMinimoZ: {indiceMinimoZ}')
    # print(f'indiceMinimo: {indiceMinimo}')
    pivote = tablero[indiceMinimo][indiceMinimoZ]

    print(pivote)
    print('letra fila: ', filasLetras[indiceMinimoZ])
    print('letra columna: ', columnasLetras[indiceMinimo])


    columnasLetras[indiceMinimo] = filasLetras[indiceMinimoZ]

    columnasLen = tablero.shape[0] -1
    if filaPivote != 0:
        columnasLen -= indiceMinimo
    
    for i in range(columnas-1):
        if pivote < epsilon:
            print('division por cero!')
        else: tablero[indiceMinimo][i] = tablero[indiceMinimo][i] / pivote 

    filasLetras.append('Operacion')
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    filasLetras.pop()

    print("indiceMinimo ", indiceMinimo)
    return tablero,indiceMinimo 

def Pivotear(tablero,filaPivote=0,columna=0):

    print('Pivotear')

    filaZ = tablero[0][:-2]
    #print(f'fila z : {filaZ}')
    valorMinimoZ = np.min(filaZ)

    
    indiceMinimoZ = np.argmin(filaZ)
    filas = tablero.shape[0]
    columnas = tablero.shape[1]

    columnaOperacion = tablero[:, -1]
    valorExcluir = -1
    print('columnaOperacion ', columnaOperacion)

    columnaOperacionLimpia = columnaOperacion[(columnaOperacion > 0) & (columnaOperacion != -1)]
    print(columnaOperacion)
    print(columnaOperacionLimpia)

    valorMinimo = np.min(columnaOperacionLimpia)

    
    indiceMinimo = np.where(columnaOperacion == valorMinimo)[0][0]

    if columna != 0:
        indiceMinimoZ = columna

    if filaPivote != 0:
        indiceMinimo = filaPivote


    print('valorMinimo: ', valorMinimo)
    print('indiceMinimo: ', indiceMinimo)

    print(f"coordendas: {indiceMinimo} ,{indiceMinimoZ} ")

    #ImprimirTabla(columnasLetras,filasLetras,tablero)
    
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
    
    ImprimirTabla(columnasLetras,filasLetras,tablero)

    aux = 0
    return tablero,aux

def Simplex(A, b, c, ci, signos, objetivo):
    global filasLetras, columnasLetras
    print("Simplex")
    AA = FormaAmpliada(A, b, c, ci, signos, objetivo, filasLetras, columnasLetras)

    #Fase 1: solo si hay  artificiales y de exceso
    aux = False
    procedimiento = ''
    menorIgual = 0
    mayorIgual = 0
    for i in range(len(signos)-len(b)):
        if signos[i] == 1:
            menorIgual += 1
        elif signos[i] == 3:
            mayorIgual += 1

    if menorIgual > 0 and mayorIgual > 0:
        procedimiento = "metodo de fases mixto"
    if menorIgual > 0 and mayorIgual == 0:
        procedimiento = 'simplex normal'
    if menorIgual == 0 and mayorIgual > 0:
        procedimiento = 'metodo de dos fases normal'

    if procedimiento == "simplex normal":
        aux = True

    auxValor = 0
    while aux:
        AA,auxValor = Encontrar_col_pivote(AA)
        AA,auxValor = Encontrar_fila_pivote(AA)
        AA,auxValor = Pivotear(AA)
        
        zFuncion = AA[0, 1:-1]  
        aux = np.any(zFuncion < 0) 

    return AA

def ImprimirTabla(tablero, filasLetras, columnasLetras):
    print('Datos que recibe la función ImprimirTabla:')
    print('Filas:', filasLetras)
    print('Columnas:', columnasLetras)

    # Agregar etiqueta de fila al inicio de cada fila de tablero
    datosFilas = [[fila] + list(valores) for fila, valores in zip(filasLetras, tablero)]

    # Encabezado de la tabla
    encabezado = [''] + columnasLetras

    tabla = tabulate(datosFilas, headers=encabezado, tablefmt="grid")
    print(tabla)

#def parser(restriccion_str):
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

#def parse_obj(obj_str):
    x1, x2 = sp.symbols('x1 x2')
    expresion = sp.sympify(obj_str)
    coeficiente1 = expresion.coeff(x1, 1)
    coeficiente2 = expresion.coeff(x2, 1)
    return [float(coeficiente1), float(coeficiente2)]

#def menu():

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

    A = np.vstack([A,[1,0]])
    A = np.vstack([A,[0,1]])

    b = np.vstack([b,[0]])
    b = np.vstack([b,[0]])

    signos = np.vstack([signos,[3]])
    signos = np.vstack([signos,[3]])

    # 2) Leer función objetivo
    print("\nAhora ingresa la función objetivo, ej: 30000*x1 + 4000*x2")
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
    print("\nVector c (coeficientes de la función objetivo):")
    print(c)
    Simplex(A,b,c,ci,signos,'maximizar')
    return 

#menu()

A = np.array([[1,0],
              [0,2],
              [3,2],])

b = np.array([[4],
              [12],
              [18]])

#Porque es negativo?
c = np.array([[30000],[50000]])
#signos
# < 0
# <= 1
# > 2
# >= 3
# != 4
sign = np.array([[1],[1],[1],[3],[3]])

ci = np.array([[0]])

simplexResultado = Simplex(A,b,c,ci,sign,'maximizar')
