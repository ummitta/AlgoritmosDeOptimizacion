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

    print("Varibles tamaño columna: ", np.shape(A)[1])

    for i in range(np.shape(variables)[0]):

        for j in range(np.shape(b)[0]-1):
            #print(variables)
            
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

    print("zfila: \n", zFila)
    print("Variables: \n", variables)
    
    

    #Creacion columna LD
    ldColumna = np.concatenate((ci,b),axis = 0)



    zFilaVariables = np.concatenate((zFila,variables),axis=0)


    zColumnaZFilaVariables = np.concatenate((zColumna,zFilaVariables),axis=1)
    #print(zColumnaZFilaVariables)

    tablero = np.concatenate((zColumnaZFilaVariables,ldColumna),axis=1)


    if cantidadHolguras > 0 and cantidadArtificiales == 0:
        for i in range(cantidadHolguras):
            print(i)

            holgura = 'h'+ str(i+1)
            filasLetras.append(holgura)
            columnasLetras.append(holgura)
    if cantidadArtificiales > 0 and cantidadHolguras == 0:
        for i in range(cantidadArtificiales):
            print(i)

            artificial = 'a'+ str(i+1)
            exceso = 'e'+str(i+1)
            filasLetras.append(exceso)
            filasLetras.append(artificial)
            columnasLetras.append(artificial)

    if cantidadArtificiales > 0 and cantidadHolguras > 0:
        for i in range(cantidadArtificiales):
            print(i)

            artificial = 'a'+ str(i+1)
            exceso = 'e'+str(i+1)
            filasLetras.append(exceso)
            filasLetras.append(artificial)
            columnasLetras.append(artificial)
        
        for i in range(cantidadHolguras):
            print(i)

            holgura = 'h'+ str(i+1)
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
    print('valor minimo',valorMinimo)
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
        if pivote < epsilon:
            print('division por cero!')
        else: tablero[indiceMinimo][i] = tablero[indiceMinimo][i] / pivote 

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

    
    return tablero

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
    print('Tablero con zeros', indices)
    # Restar combinación lineal de las variables básicas
    for i in range(len(c1)):
        zFila[i+1] = c1[i][0]

    tablero[0] = zFila
    tablero[0][0] = -1
    #while (tablero[0][1] > 0) or (tablero[0][2] > 0):
     #   tablero[0][1] = tablero[0][1] + (tablero[0][1] * -1)
     #   tablero[0][2] = tablero[0][2] + (tablero[0][2] * -1)

    
    # aplicar reduccion gaussiana para hacer 0 los coeficientes x1 y x2
    ImprimirTabla(columnasLetras, filasLetras,tablero)
    AjustarFilaZ(tablero, columnasLetras, filasLetras)
    ImprimirTabla(columnasLetras, filasLetras,tablero)
        
    return tablero

def AjustarFilaZ(tablero, columnasLetras, filasLetras):

    # Primero, identificamos las variables básicas en la solución actual
    for j in range(1, tablero.shape[1]):
        if filasLetras[j] in columnasLetras:  # Si la columna j corresponde a una variable básica
            if filasLetras[j].startswith('x'):
                fila_basica = columnasLetras.index(filasLetras[j])  # Índice de la fila de la variable básica
                print('fila_basica:',fila_basica)
            tablero[0] += tablero[fila_basica] * tablero[0, j] * -1  # Actualizar RHS (última columna)

    return tablero

def Metodo_dos_fases_Mixto(tablero,columnasLetras,filasLetras,signos):
    print("Comienzo fase dos mixto")
    ImprimirTabla(columnasLetras,filasLetras,tablero)   
    
    # cantidadRestricciones = np.shape(tablero)[0] -1
    # cantidadArtificial = np.count_nonzero(signos == 3)
    # cantidadRestriccionesArtificiales = cantidadArtificial - cantidadRestricciones
    # print(cantidadRestriccionesArtificiales)

    # for i in range(1,cantidadRestriccionesArtificiales+1):

    #     tablero[i] *= -1
    

    
    #tablero = Metodo_dos_fases_fase_1(tablero,columnasLetras,filasLetras)
    
    #creacion de coordenadas de las varaibles artificiales

    # artificialesFilas = []
    # artificialesColumna = []
    # cantidadCocientes = 0
    # for i in range(0,len(columnasLetras)):
    #     print('letras: ',columnasLetras[i])
    #     if columnasLetras[i][0] == 'a':
    #         artificialesFilas.append(i)


    # for i in range(0,len(filasLetras)):
    #     if filasLetras[i][0] == 'a':
    #         artificialesColumna.append(i)

    # print(artificialesFilas)
    # print(artificialesColumna)
    cantidadCocientes = 0


    cocientesFuncion = 0
    for letra in filasLetras:

        if letra.startswith('x'):
            cocientesFuncion += 1



    zfuncion = np.zeros((1, len(tablero[0])), dtype=float)

    for i in range(1,cocientesFuncion+1):
        zfuncion[0][i] = tablero[0][i]
        tablero[0][i] = 0
  

    print("fila z: ", zfuncion)



    artificialesFilas = []
    artificialesColumna = []
    for i in range(0,len(columnasLetras)):
        print('letras: ',columnasLetras[i])
        if columnasLetras[i][0] == 'a':
            artificialesFilas.append(i)

    for i in range(0,len(filasLetras)):
        if filasLetras[i][0] == 'a':
            artificialesColumna.append(i)

    print(artificialesFilas)
    print(artificialesColumna)

    for i in range(1,len(artificialesFilas)+1):
        for j in range(1,np.shape(tablero)[1]):
            numero = tablero[0][j]
            pivote = tablero[i][j]
            print(f"operacion: {numero} + ({pivote} * -1)")
            
            tablero[0][j] = numero + (pivote * -1)
            print(i,j)


    ImprimirTabla(columnasLetras,filasLetras,tablero)


    #Comparador para decidir cual columna usar si todos son iguales
    todoIguales = True
    filaCocientes = tablero[0,1:cocientesFuncion+1]
    print(filaCocientes)
    for i in range(1,cocientesFuncion+1):
        aux = tablero[0][i]

        for j in range(0,cocientesFuncion):
            #print(filaCocientes[j])
            if aux != filaCocientes[j]:
                 todoIguales = False

        
    print("iguales ", i)
    print("Todos iguales ", todoIguales)
        #if zfuncion[0][i] 

    todoIguales = True
    print(zfuncion)
    columnaPivote = 0

    if todoIguales:
        for i in range(1,len()):
            ...
        
    #Recuerda ahora creaer un pivote,fila, y coso solamente para si son iwales


    # tablero = Encontrar_col_pivote(tablero)      
    # tablero = Encontrar_fila_pivote(tablero)
    # print("Pivoteo Terminao")
    # tablero = Pivotear(tablero)

    # tablero = Encontrar_col_pivote(tablero)      
    # tablero = Encontrar_fila_pivote(tablero)
    # print("Pivoteo Terminao")
    # tablero = Pivotear(tablero)



    # tablero = np.delete(tablero,artificialesColumna,axis=1)
    
    # filasLetras = [valor for i, valor in enumerate(filasLetras) if i not in artificialesColumna]

    # #print(tablero)
    # ImprimirTabla(columnasLetras,filasLetras,tablero)

    # print(np.shape(tablero)[1])
    # print("fila z: ", zfuncion)
    # for i in range(1,np.shape(tablero)[1]):
    #     tablero[0][i] = zfuncion[0][i]

    # print("TABLA ANTES DEL SIMPLEX NORMAL")
    # ImprimirTabla(columnasLetras,filasLetras,tablero)
    # tablero = Encontrar_fila_pivote(tablero)
    # tablero = Encontrar_col_pivote(tablero)    
    # tablero = Pivotear(tablero)

    # tablero = Encontrar_fila_pivote(tablero)
    # tablero = Encontrar_col_pivote(tablero)    
    # tablero = Pivotear(tablero)

    # tablero = Encontrar_fila_pivote(tablero)
    # tablero = Encontrar_col_pivote(tablero)    
    # tablero = Pivotear(tablero)

    # tablero = Encontrar_fila_pivote(tablero)
    # tablero = Encontrar_col_pivote(tablero)    
    # tablero = Pivotear(tablero)
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
        AA = Metodo_dos_fases_Mixto(AA,columnasLetras,filasLetras,signos)

    if procedimiento == "simplex normal":
        aux = True

    while aux:
        AA = Encontrar_col_pivote(AA)
        AA = Encontrar_fila_pivote(AA)
        AA = Pivotear(AA)
        
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

print("####################################")
print("""

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣤⣤⣤⣤⣶⣦⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡿⠛⠉⠙⠛⠛⠛⠛⠻⢿⣿⣷⣤⡀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⠋⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠈⢻⣿⣿⡄⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⣸⣿⡏⠀⠀⠀⣠⣶⣾⣿⣿⣿⠿⠿⠿⢿⣿⣿⣿⣄⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⣿⣿⠁⠀⠀⢰⣿⣿⣯⠁⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣷⡄⠀ 
⠀⠀⣀⣤⣴⣶⣶⣿⡟⠀⠀⠀⢸⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⠀ 
⠀⢰⣿⡟⠋⠉⣹⣿⡇⠀⠀⠀⠘⣿⣿⣿⣿⣷⣦⣤⣤⣤⣶⣶⣶⣶⣿⣿⣿⠀ 
⠀⢸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀ 
⠀⣸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠉⠻⠿⣿⣿⣿⣿⡿⠿⠿⠛⢻⣿⡇⠀⠀ 
⠀⣿⣿⠁⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣧⠀⠀ 
⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀ 
⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀ 
⠀⢿⣿⡆⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀ 
⠀⠸⣿⣧⡀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⠃⠀⠀ 
⠀⠀⠛⢿⣿⣿⣿⣿⣇⠀⠀⠀⠀⠀⣰⣿⣿⣷⣶⣶⣶⣶⠶⠀⢠⣿⣿⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⣽⣿⡏⠁⠀⠀⢸⣿⡇⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⢹⣿⡆⠀⠀⠀⣸⣿⠇⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⢿⣿⣦⣄⣀⣠⣴⣿⣿⠁⠀⠈⠻⣿⣿⣿⣿⡿⠏⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠈⠛⠻⠿⠿⠿⠿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀


""")




#EJEMPLO DE MINIMIZAR 

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

print("####################################")
print("""

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣤⣤⣤⣤⣶⣦⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡿⠛⠉⠙⠛⠛⠛⠛⠻⢿⣿⣷⣤⡀⠀⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⠋⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠈⢻⣿⣿⡄⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⣸⣿⡏⠀⠀⠀⣠⣶⣾⣿⣿⣿⠿⠿⠿⢿⣿⣿⣿⣄⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⣿⣿⠁⠀⠀⢰⣿⣿⣯⠁⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣷⡄⠀ 
⠀⠀⣀⣤⣴⣶⣶⣿⡟⠀⠀⠀⢸⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⠀ 
⠀⢰⣿⡟⠋⠉⣹⣿⡇⠀⠀⠀⠘⣿⣿⣿⣿⣷⣦⣤⣤⣤⣶⣶⣶⣶⣿⣿⣿⠀ 
⠀⢸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀ 
⠀⣸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠉⠻⠿⣿⣿⣿⣿⡿⠿⠿⠛⢻⣿⡇⠀⠀ 
⠀⣿⣿⠁⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣧⠀⠀ 
⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀ 
⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀ 
⠀⢿⣿⡆⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡇⠀⠀ 
⠀⠸⣿⣧⡀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⠃⠀⠀ 
⠀⠀⠛⢿⣿⣿⣿⣿⣇⠀⠀⠀⠀⠀⣰⣿⣿⣷⣶⣶⣶⣶⠶⠀⢠⣿⣿⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⣽⣿⡏⠁⠀⠀⢸⣿⡇⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⢹⣿⡆⠀⠀⠀⣸⣿⠇⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⢿⣿⣦⣄⣀⣠⣴⣿⣿⠁⠀⠈⠻⣿⣿⣿⣿⡿⠏⠀⠀⠀⠀ 
⠀⠀⠀⠀⠀⠀⠀⠈⠛⠻⠿⠿⠿⠿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀


""")

# # EJEMPLO DE Maximizar 

# columnasLetras = ['Z']
# filasLetras = ['Z','x1','x2','x3','x4']

# A2 = np.array([[1,0,1,0],
#               [0,1,0,1],
#               [2,-1,2,-1],
#               [1,1,0,0],
#               [0,0,1,1]])

# b2 = np.array([[40],
#               [70],
#               [0],
#               [180],
#               [45]])

# #Porque es negativo?
# c2 = np.array([[1500],[1400],[1600],[1450]])
# #signos
# # < 0
# # <= 1
# # > 2
# # >= 3
# # != 4
# sign2 = np.array([[3],[3],[1],[1],[1],[3],[3],[3],[3]])

# ci2 = np.array([[0]])

# simplexResultado = Simplex(A2,b2,c2,ci2,sign2,'maximizar')

