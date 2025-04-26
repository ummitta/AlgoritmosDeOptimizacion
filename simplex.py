import numpy as np

import matplotlib.pyplot as plt

from tabulate import tabulate


np.set_printoptions(suppress=True, precision=2)

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

    #Si hay algun >= se vuelven ceros los coeficientes

    if cantidadArtificiales == 0:
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

            holgura = 'a'+ str(i+1)
            filasLetras.append(holgura)
            columnasLetras.append(holgura)
    

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
        print(f'{numerador} / {denominador}')

        if denominador != 0.0:
            operatoria[i][0] = numerador / denominador
        else:
            operatoria[i][0] = 666
            
        
    tablero = np.concatenate((tablero,operatoria),axis=1)
    
    filasLetras.append('Operacion')
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    filasLetras.pop()
    return tablero


def Encontrar_fila_pivote(tablero):

    print('Encontrar_fila_pivote')

    filaZ = tablero[0]
    valorMinimoZ = np.min(filaZ)
    indiceMinimoZ = np.argmin(filaZ)
    filas = tablero.shape[0]
    columnas = tablero.shape[1]

    columnaOperacion = tablero[:, -1]
    valorExcluir = 666
    columnaOperacionFiltrada = columnaOperacion[columnaOperacion != valorExcluir]
    columnaOperacionFiltrada = np.delete(columnaOperacionFiltrada,0,axis=0)

    valorMinimo = np.min(columnaOperacionFiltrada)

    indiceMinimo = np.where(columnaOperacion == valorMinimo)[0][0]

    #print(indiceMinimoZ,indiceMinimo)
    pivote = tablero[indiceMinimo][indiceMinimoZ]

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

    
    filaZ = tablero[0]
    valorMinimoZ = np.min(filaZ)
    indiceMinimoZ = np.argmin(filaZ)
    filas = tablero.shape[0]
    columnas = tablero.shape[1]

    columnaOperacion = tablero[:, -1]
    valorExcluir = 666
    columnaOperacionFiltrada = columnaOperacion[columnaOperacion != valorExcluir]
    columnaOperacionFiltrada = np.delete(columnaOperacionFiltrada,0,axis=0)

    valorMinimo = np.min(columnaOperacionFiltrada)

    indiceMinimo = np.where(columnaOperacion == valorMinimo)[0][0]

    
    
    for i in range(filas):
        pivoteColumna = tablero[i][indiceMinimoZ]
        for j in range(columnas-1):

        
            if i == indiceMinimo:
                break


            numero = tablero[i][j]
            pivote = tablero[indiceMinimo][j]

            operacion = numero - (pivoteColumna * pivote)
            #print(f'operacion: {numero} - ({pivoteColumna} * {pivote} ) = {operacion}')
            tablero[i][j] = operacion

    tablero = np.delete(tablero, -1, axis=1)
    
    ImprimirTabla(columnasLetras,filasLetras,tablero)

    
    return tablero


def Simplex(A,b,c,ci,signos,objetivo):
    print("Simplex")
    AA = FormaAmpliada(A,b,c,ci,signos,objetivo)

    aux = True

    while aux:
        AA = Encontrar_col_pivote(AA)
        AA = Encontrar_fila_pivote(AA)
        AA = Pivotear(AA)
        
        zFuncion = AA[0, 1:-1]  # Excluye Z y LD
        numerosNegativos = np.any(zFuncion < 0)
        aux = numerosNegativos  # Actualiza aux correctamente

    return AA


def ImprimirTabla(columna,filas,tablero):

    datosFilas = [[fila] + list(fila_valores) for fila, fila_valores in zip(columna, tablero)] 

    encabezado = [''] + filas

    tabla = tabulate(datosFilas, headers=encabezado, tablefmt="grid")

    print(tabla)



simplexResultado = Simplex(A,b,c,ci,sign,'maximizar')
