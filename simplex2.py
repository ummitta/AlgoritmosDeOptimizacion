import numpy as np

import matplotlib.pyplot as plt

from tabulate import tabulate

import sys

epsilon = sys.float_info.epsilon

np.set_printoptions(suppress=True, precision=2)

columnasLetras = ['Z']
filasLetras = ['Z','x1','x2']

def FormaAmpliada(A,b,c,ci,signos,objetivo):

    global filas, columnas

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

    cantidadCoeficientes = np.shape(A)[0]-1
    
    variables = np.zeros((np.shape(b)[0], cantidadCoeficientes + cantidadHolgurasArtificiales))
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

    #Creacion de la fila funcion

    zFila = np.zeros((1,c.shape[0]),dtype=float)


    #Si hay algun >= se vuelven ceros los coeficientes (se va a poner en cero en el metodo de dos fases)

    # if cantidadArtificiales == 0:
    for i in range(c.shape[0]):
        zFila[0][i] = c[i]
    

    for i in range(0,variables.shape[0]):
        for j in range(cantidadCoeficientes,variables.shape[1]):
            
            

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

    tablero = np.concatenate((zColumnaZFilaVariables,ldColumna),axis=1)


    if cantidadHolguras > 0 and cantidadArtificiales == 0:
        for i in range(cantidadHolguras):
           
            holgura = 'h'+ str(i+1)
            filasLetras.append(holgura)
            columnasLetras.append(holgura)
    if cantidadArtificiales > 0 and cantidadHolguras == 0:
        for i in range(cantidadArtificiales):
         

            artificial = 'a'+ str(i+1)
            exceso = 'e'+str(i+1)
            filasLetras.append(exceso)
            filasLetras.append(artificial)
            columnasLetras.append(artificial)

    if cantidadArtificiales > 0 and cantidadHolguras > 0:
        for i in range(cantidadArtificiales):
         

            artificial = 'a'+ str(i+1)
            exceso = 'e'+str(i+1)
            filasLetras.append(exceso)
            filasLetras.append(artificial)
            columnasLetras.append(artificial)
        
        for i in range(cantidadHolguras):
            

            holgura = 'h'+ str(i+1)
            filasLetras.append(holgura)
            columnasLetras.append(holgura)

    

    filasLetras.append('LD')
 
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    
    return tablero

def Encontrar_col_pivote(tablero,posicionColumna=0):
    global filas
    global columnas
    print("Encontrar_col_pivote")

    filaZ = tablero[0, 1:-1]
    valorMinimo = np.min(filaZ)
    indiceMinimo = np.argmin(filaZ) + 1

    if posicionColumna == 0:
        print("posicion columna es igual a cero")
        filaZ = tablero[0, 1:-1]
        valorMinimo = np.min(filaZ)
        indiceMinimo = np.argmin(filaZ) + 1
    

        

    if posicionColumna != 0:
        print("posicion columna distinta de cero")
        indiceMinimo = posicionColumna
    

    filas = tablero.shape[0]
    operatoria = np.zeros((filas,1),dtype=float)
    columna = tablero.shape[1]
    indiceMinimo = int(indiceMinimo)

    print("indiceMinimo: ",indiceMinimo)
    for i in range(1,filas):
        numerador = tablero[i][columna-1]
        denominador = tablero[i][indiceMinimo]
        if posicionColumna != 0:
            denominador = tablero[i][indiceMinimo+1]
        operacion = numerador / denominador
        print(f"operacion = {numerador} / {denominador} = {operacion}")
        #no borrar
        # if operacion > 0:
        #      print(f'{numerador} / {denominador} = {numerador / denominador}')

        
        if denominador != 0.0:
            operacion = numerador / denominador
            if operacion > 0:
                operatoria[i][0] = operacion 
        else:
            operatoria[i][0] = -1

  
    indexPositivo = np.where(operatoria > 0)[0]

    valorPositivos = operatoria[indexPositivo,0]
    valorMinimoMayor = valorPositivos[np.argmin(valorPositivos)]
    indiceFila = np.where(operatoria == valorMinimoMayor)[0]
    indiceFila = indiceFila[0]

      
        
    tablero = np.concatenate((tablero,operatoria),axis=1)
    
    filasLetras.append('Operacion')
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    filasLetras.pop()
    return tablero,indiceFila

def Encontrar_fila_pivote(tablero,filaPivote=0):

    print('Encontrar_fila_pivote')


    filaZ = tablero[0][:-2]
    valorMinimoZ = np.min(filaZ)
    print(valorMinimoZ)
    indiceMinimoZ = np.argmin(filaZ)

    indiceMinimo = 0

    filas = tablero.shape[0]
    columnas = tablero.shape[1]

    if filaPivote == 0:
            
            columnaOperacion = tablero[:, -1]

            valorExcluir = -1
            columnaOperacionFiltrada = columnaOperacion[columnaOperacion != valorExcluir]
            columnaOperacionFiltrada = np.delete(columnaOperacionFiltrada,0,axis=0)
            columnaOperacionFiltrada = columnaOperacionFiltrada[columnaOperacionFiltrada > 0]
            valorMinimo = np.min(columnaOperacionFiltrada)

            indiceMinimo = np.where(columnaOperacion == valorMinimo)[0][0]


    if filaPivote != 0:
        print("filaPivote != 0")
        indiceMinimo = filaPivote
        indiceMinimoZ = filaPivote



    
    # print(f'indiceMinimoZ: {indiceMinimoZ}')
    # print(f'indiceMinimo: {indiceMinimo}')
    pivote = tablero[indiceMinimo][indiceMinimoZ]


    # print("coordenadas letras: ", indiceMinimo,indiceMinimoZ)

    # print("filasLetras: ", filasLetras)
    # print("columnasLetras: ",columnasLetras)

    # print("filasLetras[indiceMinimoZ]: ", filasLetras[indiceMinimoZ])
    # print("columnasLetras[indiceMinimo]: ",columnasLetras[indiceMinimo])

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

    valorMinimoZ = np.min(filaZ)

    
    indiceMinimoZ = np.argmin(filaZ)
    filas = tablero.shape[0]
    columnas = tablero.shape[1]

    columnaOperacion = tablero[:, -1]
    valorExcluir = -1


    columnaOperacionLimpia = columnaOperacion[(columnaOperacion > 0) & (columnaOperacion != -1)]


    valorMinimo = np.min(columnaOperacionLimpia)

    
    indiceMinimo = np.where(columnaOperacion == valorMinimo)[0][0]

    if columna != 0:
        indiceMinimoZ = columna

    if filaPivote != 0:
        indiceMinimo = filaPivote



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
        
            #Casi cero
            if abs(operacion) <= epsilon:
                tablero[i][j] = 0
            else:
                tablero[i][j] = operacion
    tablero = np.delete(tablero, -1, axis=1)
    
    ImprimirTabla(columnasLetras,filasLetras,tablero)

    aux = 0
    return tablero,aux

def Metodo_dos_fases_fase_1(tablero,columnasLetras,filasLetras):
    print("Fase 1 de metodo de dos fases")
    global filas
    global columnas
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




    #Elinando las variables basicas


    #creacion de coordenadas de las varaibles artificiales
    artificialesFilas = []

    for i in range(0,len(columnasLetras)):

        if columnasLetras[i][0] == 'a':
            artificialesFilas.append(i)



    #Eliminar las varaibles artificiale de la funcion z

    for i in range(0,tablero.shape[0]):

        for j in range(1,tablero.shape[1]):

            if i in artificialesFilas:
                valorNumero = tablero[0][j] 
                valorPivote = tablero[i][j]  
                operacion = valorNumero + (valorPivote * -1)
              
        
                tablero[0][j] = operacion
    auxValor = 0
    aux = True
    while aux:

        tablero,auxValor = Encontrar_col_pivote(tablero)
        tablero,auxValor = Encontrar_fila_pivote(tablero)
    
        tablero,auxValor = Pivotear(tablero)

        zFila = tablero[0, 1:-1]
        numerosNegativos = np.any(zFila < 0)
    
        aux = numerosNegativos
    return tablero

def Metodo_dos_fases_fase_2(tablero):
    global filasLetras
    global columnasLetras
    global filas
    global columnas
    print('comienza fase2')
    # Eliminar columnas de variables artificiales
    indices = [i for i, letra in enumerate(filasLetras) if letra.startswith('a')]
    print('Indices que voy a borrar:', indices)

    if indices:  # Solo si hay columnas a borrar
        tablero = np.delete(tablero, indices, axis=1)  # Eliminar todas las columnas artificiales de una sola
        filasLetras = [letra for i, letra in enumerate(filasLetras) if i not in indices]

    # Ahora reconstruir la fila Z (función objetivo)
    
    zFila = np.zeros(tablero.shape[1])

    # Restar combinación lineal de las variables básicas
    for i in range(len(c1)):
        zFila[i+1] = c1[i][0]

    tablero[0] = zFila
    tablero[0][0] = -1
    #while (tablero[0][1] > 0) or (tablero[0][2] > 0):
     #   tablero[0][1] = tablero[0][1] + (tablero[0][1] * -1)
     #   tablero[0][2] = tablero[0][2] + (tablero[0][2] * -1)

    
    # aplicar reduccion gaussiana para hacer 0 los coeficientes x1 y x2

    AjustarFilaZ(tablero, columnasLetras, filasLetras)
    ImprimirTabla(columnasLetras, filasLetras,tablero)
        
    return tablero

def AjustarFilaZ(tablero, columnasLetras, filasLetras):

    # Primero, identificamos las variables básicas en la solución actual
    for j in range(1, tablero.shape[1]):
        if filasLetras[j] in columnasLetras:  # Si la columna j corresponde a una variable básica
            if filasLetras[j].startswith('x'):
                fila_basica = columnasLetras.index(filasLetras[j])  # Índice de la fila de la variable básica
              
            tablero[0] += tablero[fila_basica] * tablero[0, j] * -1  # Actualizar RHS (última columna)

    return tablero

def Metodo_dos_fases_Mixto_fase1(tablero,columnasLetras,filasLetras,signos):
    print("Comienzo fase dos mixto")

    cantidadCocientes = 0


    cocientesFuncion = 0
    for letra in filasLetras:

        if letra.startswith('x'):
            cocientesFuncion += 1



    zfuncion = np.zeros((1, len(tablero[0])), dtype=float)

    for i in range(1,cocientesFuncion+1):
        zfuncion[0][i] = tablero[0][i]
        tablero[0][i] = 0
  





    artificialesFilas = []
    artificialesColumna = []
    for i in range(0,len(columnasLetras)):
  
        if columnasLetras[i][0] == 'a':
            artificialesFilas.append(i)

    for i in range(0,len(filasLetras)):
        if filasLetras[i][0] == 'a':
            artificialesColumna.append(i)



    for i in range(1,len(artificialesFilas)+1):
        for j in range(1,np.shape(tablero)[1]):
            numero = tablero[0][j]
            pivote = tablero[i][j]
        
            
            tablero[0][j] = numero + (pivote * -1)
    



    aux = True
    indiceColumna = 0
    filaPivote = 0

    print("While metodo dos fases mixto fase 1")
    while aux:

        todoIguales, columnaPivote = ValoresFuncionIguales(tablero)
        valoresNegativos = BuscarValoresNegativos(tablero)


        if todoIguales == True:
            tablero,indiceColumna = Encontrar_col_pivote(tablero,columnaPivote)
            tablero,filaPivote = Encontrar_fila_pivote(tablero,indiceColumna)
            tablero,aux = Pivotear(tablero,filaPivote,indiceColumna)


            zFila = tablero[0, 1:-1]
            aux = np.any(zFila < 0)
            print(zFila)


            continue

        tablero, indiceColumna = Encontrar_col_pivote(tablero)

        tablero,filaPivote = Encontrar_fila_pivote(tablero)

        tablero,aux = Pivotear(tablero)

        zFila = tablero[0, 1:-1]
    
        aux = np.any(zFila < 0)


    print("final fase mixta")
    ImprimirTabla(columnasLetras,filasLetras,tablero)
    tablero[0] = zfuncion
    return tablero


def ValoresFuncionIguales(tablero):



    zfuncion = np.zeros((1, len(tablero[0])), dtype=float)

    cocientesFuncion = 1
    for letra in filasLetras:

        if letra.startswith('x'):
            cocientesFuncion += 1



    for i in range(1,cocientesFuncion+1):
        zfuncion[0][i] = tablero[0][i]
        


    # todoIguales = np.all(zfuncion[:cocientesFuncion] == zfuncion[0][1])
 

    todoIguales = True

    for i in range(1,cocientesFuncion):

        if tablero[0][i] != zfuncion[0][1]:
   
            todoIguales = False
        if tablero[0][i] == 0:
            todoIguales = False





    columnaPivote = tablero[0][1]


    if todoIguales:
        for i in range(1,np.shape(tablero)[1]-1):
          

            if tablero[0][i] > columnaPivote:
                columnaPivote = tablero[0][i]
    
    else:
        columnaPivote = 0
    
    columnaPivote = int(columnaPivote)
    ImprimirTabla(columnasLetras,filasLetras,tablero)



    return todoIguales, columnaPivote

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




def EliminarColumnasArtificiales(tablero):

    print("Eliminar Columnas Artificiales")


    artificialesColumna = []
    for i in range(0,len(filasLetras)):
         if filasLetras[i][0] == 'a':
             artificialesColumna.append(i) 



    tablero = np.delete(tablero, artificialesColumna,axis=1)


    return tablero


def Metodo_dos_fases_Mixto_fase2(tablero,columnasLetras=0,filasLetras=0):

    print("Metodo_dos_fases_Mixto_fase2 ")

    tablero = EliminarColumnasArtificiales(tablero)
    #tablero, filaPivote = Encontrar_fila_pivote(tablero)

    print("Despues de eliminar las columnas artificiales")
    ImprimirTabla(columnasLetras,filasLetras,tablero)

    aux = True
    while aux:

        tablero,auxValor = Encontrar_col_pivote(tablero)
        tablero,auxValor = Encontrar_fila_pivote(tablero)
    
        tablero,auxValor = Pivotear(tablero)

        zFila = tablero[0, 1:-1]
        numerosNegativos = np.any(zFila < 0)

        aux = numerosNegativos
    return tablero


    

def Simplex(A, b, c, ci, signos, objetivo):
    print("Simplex")
    AA = FormaAmpliada(A, b, c, ci, signos, objetivo)

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

    if procedimiento == 'metodo de dos fases normal':
        AA = Metodo_dos_fases_fase_1(AA, columnasLetras, filasLetras)
        AA = Metodo_dos_fases_fase_2(AA)

    if procedimiento == "metodo de fases mixto":
        AA = Metodo_dos_fases_Mixto_fase1(AA,columnasLetras,filasLetras,signos)

        AA =  Metodo_dos_fases_Mixto_fase2(AA,columnasLetras,filasLetras)
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


def ImprimirTabla(columna,filas,tablero):

    datosFilas = [[fila] + list(fila_valores) for fila, fila_valores in zip(columna, tablero)] 

    encabezado = [''] + filas

    tabla = tabulate(datosFilas, headers=encabezado, tablefmt="grid")

    print(tabla)


#EJEMPLO DE MAXIMIZAR 

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

print("###########################################")
print("###########################################")
print("#           PROBLEMA SIGUIENTE           #")
print("###########################################")
print("###########################################")


# #EJEMPLO DE MINIMIZAR 

columnasLetras = ['Z']
filasLetras = ['Z','x1','x2']

A1 = np.array([[60,60],
              [12,6],
              [10,30]])

b1 = np.array([[300],
              [36],
              [90]])

#Porque es negativo?
c1 = np.array([[0.12],[0.15]])
#signos
# < 0
# <= 1
# > 2
# >= 3
# != 4
sign1 = np.array([[3],[3],[3],[3],[3]])

ci1 = np.array([[0]])

simplexResultado = Simplex(A1,b1,c1,ci1,sign1,'minimizar')

print("###########################################")
print("###########################################")
print("#           PROBLEMA SIGUIENTE           #")
print("###########################################")
print("###########################################")

# # EJEMPLO DE Maximizar 

columnasLetras = ['Z']
filasLetras = ['Z','x1','x2','x3','x4']

A2 = np.array([[1,0,1,0],
              [0,1,0,1],
              [2,-1,2,-1],
              [1,1,0,0],
              [0,0,1,1]])

b2 = np.array([[40],
              [70],
              [0],
              [180],
              [45]])

#Porque es negativo?
c2 = np.array([[1500],[1400],[1600],[1450]])
#signos
# < 0
# <= 1
# > 2
# >= 3
# != 4
sign2 = np.array([[3],[3],[1],[1],[1],[3],[3],[3],[3]])

ci2 = np.array([[0]])

simplexResultado = Simplex(A2,b2,c2,ci2,sign2,'maximizar')

