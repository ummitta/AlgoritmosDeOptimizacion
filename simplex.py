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



# def FormaAmpliada(A,b,c,ci,signos,objetivo):
#     zz = 0
#     if objetivo == "maximizar":
#         zz = 1
#     elif objetivo == "minimizar":
#         zz = -1

#     c =  -1 *c

#     print('A')
#     print(A)
#     print('b')
#     print(b)
#     print('c')
#     print(c)
#     print('ci')
#     print(ci)
#     print('objetivo')
#     print(objetivo)

#     cc = np.reshape(c,(1,2))
#     zeros = np.zeros((1,len(b)),dtype=float)
#     ccZeros = np.concatenate((cc,zeros),axis=1)
#     z = np.concatenate((ccZeros,ci),axis=1)
  

#     indentity = np.identity(np.shape(b)[0],dtype=float)

#     zc = np.zeros((indentity.shape[0],1),dtype=float)

#     zf = np.array([zz])

#     zc = np.vstack((zf,zc))

    
#     #print(np.shape(indentity))
#     AI = np.concatenate((A,indentity),axis=1)
#     AIb = np.concatenate((AI,b),axis=1)
    
#     #print(zc)
#     A_function = np.concatenate((z,AIb),axis=0)

#     A_Ampliada = np.concatenate((zc,A_function),axis=1)


#     print("TABLERO DE FORMAAMPLIDAD")
#     return A_Ampliada


def FormaAmpliada(A,b,c,ci,signos,objetivo):

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

    
    # filas = ['Z', 'h1', 'h2','h3']
    # columnas = ['Z','x1', 'x1', 'h1', 'h2', 'h3', 'LD']
    
    # ImprimirTabla()
    
    return tablero


def Encontrar_col_pivote(tablero):
    print("Encontrar_col_pivote")

    filaZ = tablero[0]
    valorMinimo = np.min(filaZ)
    indiceMinimo = np.argmin(filaZ)
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
    print(tablero)

    # zColMinimoIndice = np.argmin(tablero)
    # indiceFilas = tablero.shape[0]

    # print(zColMinimoIndice)

    # Ld = np.zeros((indiceFilas,1),dtype=float)


    # tablero = np.concatenate((tablero,Ld),axis=1)

    # indiceFilas = tablero.shape[0]
    # indiceColumna = tablero.shape[1]

    # print(tablero)

    # for i in range(indiceFilas):

    #     valorNumerador = tablero[i][indiceColumna-2]
    #     valorDenominador = tablero[i][zColMinimoIndice]

    #     if valorDenominador != 0:
    #         tablero[i][indiceColumna-1] = valorNumerador / valorDenominador
            
    #     else:
    #         print("No se puede dividor por 0")
    #         tablero[i][indiceColumna-1] = -1

        #print(valorNumerador, valorDenominador)

    #print(tablero)

    
    #print(zColMinimoIndice)
    #print(valorMinimo)
    #print(valoresColumna)


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
    for i in range(columnas-1):

        tablero[indiceMinimo][i] = tablero[indiceMinimo][i] / pivote 

    print(tablero)

    return tablero



    # print("fila")
    # print(tablero)

    # zMinimo = np.argmin(tablero)
    # print(zMinimo)
    # filas = tablero.shape[0]
    # columnas = tablero.shape[1]
    # print(filas,columnas)

    # ldNumeros = tablero[:,columnas-1]

    # ldNumeroMayores = ldNumeros[ldNumeros>0]

    # ldNumero =np.min(ldNumeroMayores)

    # print(ldNumeros,ldNumero)
    # indiceNumero = np.where(ldNumeros == ldNumero)[0][0]
    # print(indiceNumero)

    # zMinimoIndex = np.argmin(tablero[0])
  
    # valorDenominador = tablero[indiceNumero][zMinimoIndex]

    # for i in range(columnas-1):
    #     valorNumerador = tablero[indiceNumero][i]

    #     print(f'{valorNumerador} / {valorDenominador}')

    #     tablero[indiceNumero][i] = valorNumerador / valorDenominador



    # print(tablero)
    

    # return tablero




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


            if i == 2:
                break


            numero = tablero[i][j]
            pivote = tablero[indiceMinimo][j]

            operacion = numero - (pivoteColumna * pivote)
            tablero[i][j] = operacion

    tablero = np.delete(tablero, -1, axis=1)
    print(tablero)
    return tablero
            #print(i,j)
    # status = False

    # filas = tablero.shape[0]
    # columnas = tablero.shape[1]

    # zMinimoIndex = np.argmin(tablero[0])

    # ldNumeros = tablero[:,columnas-1]

    # ldNumeroMayores = ldNumeros[ldNumeros>0]

    # ldNumero =np.min(ldNumeroMayores)

    # indiceNumero = np.where(ldNumeros == ldNumero)[0][0]

    # tablero = np.delete(tablero,-1,axis=1)

    # listaParaPivotear = np.arange(filas)
    # listaParaPivotear = listaParaPivotear[listaParaPivotear != indiceNumero]


    # print(listaParaPivotear)

    

    # columnas = tablero.shape[1]

    # #print(tablero)
    # #zMinimo = tablero[0][zMinimoIndex]
    
    # print('OE')
    


    # print(indiceNumero,zMinimoIndex)
    # for i in listaParaPivotear:
    #     print(f'----i: {i}')
    #     pivoteColumna = tablero[i][zMinimoIndex]
    #     for j in range(columnas):
    #         print(f'j: {j}')
    #         pivote = tablero[indiceNumero][j]
            
    #         numero = tablero[i][j]
            
    #         print(numero)
    #         print(pivote)
    #         print(pivoteColumna)
                
                
    #         operacion = numero - (pivoteColumna * pivote )
    #         print(operacion)

    #         tablero[i][j] = operacion
            


    # print('pivotear',tablero)

    # return tablero

def Simplex(A,b,c,ci,signos,objetivo):
    print("Simplex")
    AA = FormaAmpliada(A,b,c,ci,signos,objetivo)

    filas = ['Z', 'h1', 'h2','h3']
    columnas = ['Z','x1', 'x1', 'h1', 'h2', 'h3', 'LD']

    ImprimirTabla(filas,columnas,AA)


    aux = True


    while aux != False:

    
        AA = Encontrar_col_pivote(AA)
        #print(AA)
        
        AA = Encontrar_fila_pivote(AA)

        

        AA = Pivotear(AA)
        
      
        aux = True
        
        zFuncion = AA[0, :-1]
        print(zFuncion)
        numerosNegativos = np.any(zFuncion < 0)
        print(numerosNegativos)
        
        if numerosNegativos != True:
            aux = True

    return AA


def ImprimirTabla(filas,columnas,tablero):

    datosFilas = [[fila] + list(fila_valores) for fila, fila_valores in zip(filas, tablero)] 

    encabezado = [''] + columnas

    tabla = tabulate(datosFilas, headers=encabezado, tablefmt="grid")

    print(tabla)

# filas = ['Z', 'h1', 'h2','h3']
# columnas = ['Z','x1', 'x1', 'h1', 'h2', 'h3', 'LD']


simplexResultado = Simplex(A,b,c,ci,sign,'maximizar')



# datosFilas = [[fila] + list(fila_valores) for fila, fila_valores in zip(filas, simplexResultado)] 

# encabezado = [''] + columnas

# tabla = tabulate(datosFilas, headers=encabezado, tablefmt="grid")

# print(tabla)
#print(simplexResultado)


# 1 maximinar
# -1 minimizar