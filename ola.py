import numpy as np

import matplotlib.pyplot as plt


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







def FormaAmpliada(A,b,c,ci,objetivo):

    #print("A: ",np.shape(A)," B: " ,np.shape(b), "C : ",np.shape(c),"ci: ", np.shape(ci))

    #print('A :', A)
    #print('b :',b)
    #print('c :', c)
    #print('ci :', ci)
    zz = 0
    if objetivo == "maximizar":
        zz = 1
    elif objetivo == "minimizar":
        zz = -1



    c = -1* c
    cc = np.reshape(c,(1,2))
    zeros = np.zeros((1,len(b)),dtype=int)
    ccZeros = np.concatenate((cc,zeros),axis=1)
    z = np.concatenate((ccZeros,ci),axis=1)
  

    indentity = np.identity(np.shape(b)[0],dtype=int)

    zc = np.zeros((indentity.shape[0],1),dtype=int)

    zf = np.array([zz])

    zc = np.vstack((zf,zc))

    
    #print(np.shape(indentity))
    AI = np.concatenate((A,indentity),axis=1)
    AIb = np.concatenate((AI,b),axis=1)
    
    #print(zc)
    A_function = np.concatenate((z,AIb),axis=0)

    A_Ampliada = np.concatenate((zc,A_function),axis=1)



    return A_Ampliada


def Encontrar_col_pivote(tablero):
    print("col")

    print(tablero)

    zColMinimoIndice = np.argmin(tablero)
    indiceFilas = tablero.shape[0]

    print(zColMinimoIndice)

    Ld = np.zeros((indiceFilas,1),dtype=int)


    tablero = np.concatenate((tablero,Ld),axis=1)

    indiceFilas = tablero.shape[0]
    indiceColumna = tablero.shape[1]

    print(tablero)

    for i in range(indiceFilas):

        valorNumerador = tablero[i][indiceColumna-2]
        valorDenominador = tablero[i][zColMinimoIndice]

        if valorDenominador != 0:
            tablero[i][indiceColumna-1] = valorNumerador / valorDenominador
            
        else:
            print("No se puede dividor por 0")
            tablero[i][indiceColumna-1] = -1

        #print(valorNumerador, valorDenominador)

    #print(tablero)

    
    #print(zColMinimoIndice)
    #print(valorMinimo)
    #print(valoresColumna)
    return tablero


def Encontrar_fila_pivote(tablero):
    print("fila")
    print(tablero)

    zMinimo = np.argmin(tablero)
    print(zMinimo)
    filas = tablero.shape[0]
    columnas = tablero.shape[1]
    print(filas,columnas)


    ldNumeros = tablero[:,columnas-1]
    ldNumeroMayores = ldNumeros[ldNumeros>0]
    ldNumero =np.min(ldNumeroMayores)
    print(ldNumero)

    
    for j in range(columnas):
        tablero[][j]



def Simplex(A,b,c,ci,signos,objetivo):
    print("Simplex")
    AA = FormaAmpliada(A,b,c,ci,objetivo)

    aux = False
    
    AA = Encontrar_col_pivote(AA)
    #print(AA)

    Encontrar_fila_pivote(AA)



Simplex(A,b,c,ci,sign,'maximizar')


# 1 maximinar
# -1 minimizar