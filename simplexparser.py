import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
import sympy as sp

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

    #Creacion columna LD
    ldColumna = np.concatenate((ci,b),axis = 0)

    zFila = np.zeros((1, variables.shape[1]))  # Misma cantidad de columnas que variables
    zFila[0, :c.shape[1]] = c.flatten()  # Copiar coeficientes de c
    zFilaVariables = np.concatenate((zFila,variables),axis=0)

    zColumnaZFilaVariables = np.concatenate((zColumna,zFilaVariables),axis=1)
    #print(zColumnaZFilaVariables)
    #Creacion de la fila funcion
    if zColumnaZFilaVariables.shape[0] > ldColumna.shape[0]:
        ldColumna = np.vstack([ldColumna, np.zeros((zColumnaZFilaVariables.shape[0] - ldColumna.shape[0], ldColumna.shape[1]))])

    
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
    auxValor = 0
    aux = True
    while aux:

        tablero,auxValor = Encontrar_col_pivote(tablero)
        tablero,auxValor = Encontrar_fila_pivote(tablero)
    
        tablero,auxValor = Pivotear(tablero)

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
    #ImprimirTabla(columnasLetras,filasLetras,tablero)   
    
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
            #print(f"operacion: {numero} + ({pivote} * -1)")
            
            tablero[0][j] = numero + (pivote * -1)
            print(i,j)


    print("a")
    

    print("tablero[0]: ",tablero[0])

    

    #Comparador para decidir cual columna usar si todos son iguales
    # todoIguales = True
    # filaCocientes = tablero[0,1:cocientesFuncion+1]
    # print(filaCocientes)
    # for i in range(1,cocientesFuncion+1):
    #     aux = tablero[0][i]

    #     for j in range(0,cocientesFuncion):
    #         #print(filaCocientes[j])
    #         if aux != filaCocientes[j]:
    #              todoIguales = False

        
    print("iguales ", i)
    # print("Todos iguales ", todoIguales)
        #if zfuncion[0][i] 

    todoIguales = True
    
    

    columnaPivote = tablero[0][1]

    print("columnaPivote: ", columnaPivote)
    print("tamaño[0]: ", np.shape(tablero)[0])
    if todoIguales:
        for i in range(1,np.shape(tablero)[1]-1):
            print("tablero[0][i]: ",tablero[0][i])

            if tablero[0][i] > columnaPivote:
                columnaPivote = tablero[0][i]

    print(columnaPivote) 
    ImprimirTabla(columnasLetras,filasLetras,tablero)

    tablero,indiceColumna = Encontrar_col_pivote(tablero,columnaPivote)
    # tablero = Encontrar_fila_pivote(tablero,filaPivote)
    print("test ")
    print("indiceColumna: ", indiceColumna)
    tablero,filaPivote = Encontrar_fila_pivote(tablero,indiceColumna)

    print("antes de pivotear ", tablero)
    tablero,aux = Pivotear(tablero,filaPivote,indiceColumna)
    print("test")
    print("despues de pivotear ", filaPivote,indiceColumna)

    
    tablero, indiceColumna = Encontrar_col_pivote(tablero)

    tablero,filaPivote = Encontrar_fila_pivote(tablero)

    tablero,aux = Pivotear(tablero)


    tablero, indiceColumna = Encontrar_col_pivote(tablero)

    tablero,filaPivote = Encontrar_fila_pivote(tablero)

    tablero,aux = Pivotear(tablero)

    tablero[0] = zfuncion

    ImprimirTabla(columnasLetras,filasLetras,tablero)
    
    EliminarColumnasArtificiales(tablero)
    #tablero, filaPivote = Encontrar_fila_pivote(tablero)

    aux = True
    while aux:

        tablero,auxValor = Encontrar_col_pivote(tablero)
        tablero,auxValor = Encontrar_fila_pivote(tablero)
    
        tablero,auxValor = Pivotear(tablero)

        zFila = tablero[0, 1:-1]
        numerosNegativos = np.any(zFila < 0)
        print(zFila)
        aux = numerosNegativos
    return tablero

    #tablero = Pivotear(tablero)

    # tablero = Encontrar_col_pivote(tablero)
    # tablero = Encontrar_fila_pivote(tablero)
    # tablero = Pivotear(tablero)

    # tablero = Encontrar_col_pivote(tablero)
    # tablero = Encontrar_fila_pivote(tablero)
    # tablero = Pivotear(tablero)
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

def EliminarColumnasArtificiales(tablero):

    print("Eliminar Columnas Artificiales")

    print(filasLetras)
    artificialesColumna = []
    for i in range(0,len(filasLetras)):
         if filasLetras[i][0] == 'a':
             artificialesColumna.append(i) 

    print(artificialesColumna)

    tablero = np.delete(tablero, artificialesColumna,axis=1)

    #for i in range(0,len(artificialesColumna)):
    #    tablero = np.delete(tablero,artificialesColumna[i],axis=1)


    
    ImprimirTabla(columnasLetras,filasLetras,tablero)

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
    return Simplex(A,b,c,ci,signos,'maximizar')

#simplexResultado = Simplex(A,b,c,ci,sign,'maximizar')

menu()