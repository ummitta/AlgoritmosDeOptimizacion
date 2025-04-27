import numpy as np

import matplotlib.pyplot as plt

from tabulate import tabulate

import sys

epsilon = sys.float_info.epsilon

np.set_printoptions(suppress=True, precision=2)

A = np.array([[60,60],
              [12,6],
              [10,30],])

b = np.array([[300],
              [36],
              [90]])

#Porque es negativo?
c = np.array([[0.12],[0.15]])
#signos
# < 0
# <= 1
# > 2
# >= 3
# != 4
sign = np.array([[3],[3],[3],[3],[3]])

ci = np.array([[0]])

columnasLetras = ['Z']
filasLetras = ['Z','x1','x2']

def FormaAmpliada(A,b,c,ci,signos,objetivo):

    global filas
    global columnas

    #Creacion de columna Z
    zValor = 0
    if objetivo == "maximizar":
        zValor = 1

        c = -1*c
    elif objetivo == "minimizar":
        zValor = -1


    #Creacion de columna z para ver si es maximizacion o minizacion
    
    variablesBasicas = np.zeros((b.shape[0],1),dtype=float)
    zColumna = np.vstack((zValor,variablesBasicas))
    
    
    #Creacion de matriz de coeficiente, y los posibles signos
    signosLimite = signos[:A.shape[0]]

    cantidadHolguras = np.count_nonzero(signosLimite == 1)

    cantidadArtificiales = np.count_nonzero(signosLimite == 3)

    cantidadExceso = cantidadArtificiales * 2

    cantidadHolgurasArtificiales = cantidadHolguras + cantidadExceso
    
    variables = np.zeros((np.shape(b)[0], 2 + cantidadHolgurasArtificiales))
    separacion = 2

    for i in range(np.shape(variables)[0]):

        for j in range(np.shape(b)[0]-1):
            #print(variables)
            

            variables[i][0] = A[i][0]
            variables[i][1] = A[i][1]

            if signos[i] == 1:
                variables[i][separacion] = 1

    
                separacion += 1
                break

            if signos[i] == 3:
                variables[i][separacion] = -1
                variables[i][separacion+1] = 1
                separacion += 2

                break 

    #Creacion de la fila funcion

    zFila = np.zeros((1,c.shape[0]),dtype=float)


    #Si hay algun >= se vuelven ceros los coeficientes (se va a poner en cero en el metodo de dos fases)

    # if cantidadArtificiales == 0:
    for i in range(c.shape[0]):
        zFila[0][i] = c[i]
    

    for i in range(0,variables.shape[0]):
        for j in range(2,variables.shape[1]):
            
            

            if variables[i][j] == -1 and variables[i][j+1] == 1:
                aux = np.array([[0,1]])
                zFila = np.concatenate((zFila,aux),axis = 1)
                break
    
            if variables[i][j] == 1:
                aux = np.array([[0]])

                zFila = np.concatenate((zFila,aux),axis = 1)
                break


    

    #Creacion columna LD
    ldColumna = np.concatenate((ci,b),axis = 0)



    zFilaVariables = np.concatenate((zFila,variables),axis=0)


    zColumnaZFilaVariables = np.concatenate((zColumna,zFilaVariables),axis=1)
    #print(zColumnaZFilaVariables)

    tablero = np.concatenate((zColumnaZFilaVariables,ldColumna),axis=1)


    if cantidadHolguras > 0:
        for i in range(cantidadHolguras):
            print(i)

            holgura = 'h'+ str(i+1)
            filasLetras.append(holgura)
            columnasLetras.append(holgura)
    if cantidadArtificiales > 0:
        for i in range(cantidadArtificiales):
            print(i)

            artificial = 'a'+ str(i+1)
            exceso = 'e'+str(i+1)
            filasLetras.append(exceso)
            filasLetras.append(artificial)
            columnasLetras.append(artificial)
    

    filasLetras.append('LD')
 

    
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    
    return tablero






def Encontrar_col_pivote(tablero):
    global filas
    global columnas
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
    
    filasLetras.append('Operacion')
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    filasLetras.pop()
    return tablero


def Encontrar_fila_pivote(tablero):

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

   
    filasLetras.append('Operacion')
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    filasLetras.pop()
    return tablero




def Pivotear(tablero):

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

    ImprimirTabla(columnasLetras,filasLetras,tablero)
    
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

    
    return tablero

def Metodo_dos_fases_fase_1(tablero,columnasLetras,filasLetras):
    print("Fase 1 de metodo de dos fases")

    artificiales = 0
    holguras = 0
    cocientesFuncion = 0
    for letra in filasLetras:
        if letra.startswith('a'):
            artificiales += 1

        if letra.startswith('h'):
            holguras += 1

        if letra.startswith('x'):
            cocientesFuncion += 1

    #Crear funcion W
    filaWOriginal = np.copy(tablero[0])
    filaW = np.copy(tablero[0])
    filaW[1],filaW[2] = 0,0

    tablero[0] = filaW
    print(tablero)


    print(artificiales,holguras)
    #Elinando las variables basicas
    print(filasLetras)
    print(columnasLetras)

    #creacion de coordenadas de las varaibles artificiales
    artificialesFilas = []

    for i in range(0,len(columnasLetras)):
        print('letras: ',columnasLetras[i])
        if columnasLetras[i][0] == 'a':
            artificialesFilas.append(i)
    print("Filas con artificiales:", artificialesFilas)


    #Eliminar las varaibles artificiale de la funcion z

    for i in range(0,tablero.shape[0]):

        for j in range(1,tablero.shape[1]):

            if i in artificialesFilas:
                valorNumero = tablero[0][j] 
                valorPivote = tablero[i][j]  
                operacion = valorNumero + (valorPivote * -1)
              
        
                tablero[0][j] = operacion
    aux = True
    while aux:

        tablero = Encontrar_col_pivote(tablero)
        tablero = Encontrar_fila_pivote(tablero)
    
        tablero = Pivotear(tablero)

        zFila = tablero[0, 1:-1]
        numerosNegativos = np.any(zFila < 0)
        print(zFila)
        aux = numerosNegativos
        



    




def Simplex(A,b,c,ci,signos,objetivo):
    print("Simplex")
    AA = FormaAmpliada(A,b,c,ci,signos,objetivo)


    #Fase 1: solo si hay  artificiales y de exceso
    tablero = Metodo_dos_fases_fase_1(AA,columnasLetras,filasLetras)


    aux = False
    

    while aux:
        AA = Encontrar_col_pivote(AA)
        AA = Encontrar_fila_pivote(AA)
        AA = Pivotear(AA)
        
        zFuncion = AA[0, 1:-1]  
        numerosNegativos = np.any(zFuncion < 0)
        aux = numerosNegativos  

    return AA


def ImprimirTabla(columna,filas,tablero):

    datosFilas = [[fila] + list(fila_valores) for fila, fila_valores in zip(columna, tablero)] 

    encabezado = [''] + filas

    tabla = tabulate(datosFilas, headers=encabezado, tablefmt="grid")

    print(tabla)

print(epsilon)

simplexResultado = Simplex(A,b,c,ci,sign,'minimizar')
