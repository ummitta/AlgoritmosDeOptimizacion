import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys
import sympy as sp

epsilon = sys.float_info.epsilon
np.set_printoptions(suppress=True, precision=2)

def FormaAmpliada(A, b, c, ci, signos, obj, metodo, filasLetras, columnasLetras):
    zValor = 0
    if obj == "max":
        zValor = 1
    elif obj == "min":
        zValor = -1
        c = -1*c

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
            columnasLetras.append('e' + str(i+1))
            columnasLetras.append('a' + str(i+1))
            filasLetras.append('e' + str(i+1))
            excesos += 2

    
    columnasLetras.append('LD')

    #Creacion de columna z (del largo de columnaletras)        
    zFila = np.zeros((1, len(columnasLetras)), dtype=float)

    if metodo == "2fases":
        print("metodo 2 fases")
        wFila = np.zeros_like(zFila)
        wFila[0, 0] = zValor  # W en la posición [0][0]

        for i in range(len(signos)):
            if signos[i] == 3:
                filaRestriccion = np.zeros((1, len(columnasLetras)), dtype=float)
                filaRestriccion[0, 1:1+len(A[i])] = A[i]
                filaRestriccion[0, -1] = b[i]
                
                # Agregar exceso (-1) y artificial (+1)
                indiceE = columnasLetras.index('e' + str(i+1))
                indiceA = columnasLetras.index('a' + str(i+1))
                filaRestriccion[0, indiceE] = -1
                filaRestriccion[0, indiceA] = 1

                wFila = wFila + filaRestriccion

        zFila = -1 * wFila

    if metodo == "simplex":
        zFila[0, 0] = zValor  # Z en la posición [0][0]
        for i in range(np.shape(c)[1]):
            zFila[0][i+1] = c.flatten()[i]
    
    if metodo == "mixto":
        print("metodo mixto activado")
        wFila = np.zeros_like(zFila)
        wFila[0, 0] = zValor  # Z en columna 0

        for i in range(len(signos)):
            if signos[i] == 3:  # Solo restricciones >= participan en W
                filaRestriccion = np.zeros((1, len(columnasLetras)), dtype=float)

                # Coeficientes de las variables x1, x2, ..., xn
                filaRestriccion[0, 1:1 + len(A[i])] = A[i]
                filaRestriccion[0, -1] = b[i]

                # Índices para exceso y artificial
                indiceE = columnasLetras.index('e' + str(i + 1))
                indiceA = columnasLetras.index('a' + str(i + 1))
                filaRestriccion[0, indiceE] = -1  # exceso con -1
                #filaRestriccion[0, indiceA] = 1   # artificial con +1

                # Sumar esta fila a la función W
                wFila = wFila + filaRestriccion

        # Crear Z = -W para usar como fila inicial en el tablero (fase 1 parcial)
        print("Zfila dentro de modo mixto: ", wFila)
        zFila = -1 * wFila
        print("Zfila dentro de modo mixto: ", zFila)



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
                variables[i][separacion+1] = 1
                tablero[i+1][separacion+1] = -1
                tablero[i+1][separacion+2] = 1
                separacion += 2
                break

    print("variables: \n", variables)
    print("tablero: \n", tablero)
    print("Imprimir tabla en forma ampliada, justo antes de retornar ")
    ImprimirTabla(tablero, filasLetras, columnasLetras)
    
    return tablero

def Pivoteo(tablero, filasLetras, columnasLetras):

    filas, columnas = tablero.shape
    print("Iniciando pivote para todos los casos")
    
    while True:

        filaZ = tablero[0, 1:-1]


        #extrae los negativos 
        negativos = [
            j+1 for j, val in enumerate(filaZ)
            if val < -epsilon and columnasLetras[j+1].startswith(('x', 'h', 'e'))
        ]
        if not negativos:
            print("No quedan coeficientes negativos en Z. Fin de pivoteo.")
            break
        
        j_ent = min(negativos, key=lambda j: tablero[0, j])
        print(f"Variable entrante: {columnasLetras[j_ent]} (columna {j_ent})")
        
        ratios = np.full(filas, np.inf)
        for i in range(1, filas):
            aij = tablero[i, j_ent]
            if aij > epsilon:
                ratios[i] = tablero[i, -1] / aij
        i_sal = np.argmin(ratios)
        print(f"Variable saliente: {filasLetras[i_sal]} (fila {i_sal})")
        
        filasLetras[i_sal] = columnasLetras[j_ent]
        
   
        piv = tablero[i_sal, j_ent]
        tablero[i_sal, 1:] /= piv

        for i in range(filas):
            if i == i_sal:
                continue
            factor = tablero[i, j_ent]
            tablero[i, 1:] -= factor * tablero[i_sal, 1:]
            tablero[i, np.abs(tablero[i]) <= epsilon] = 0.0
        
        print(f"Pivote aplicado en fila {i_sal}, columna {j_ent}")
        ImprimirTabla(tablero, filasLetras, columnasLetras)
    
    return tablero

def elimVarArtificiales(tablero, filasLetras, columnasLetras):
    print("Eliminar variables artificiales")

    # Identificar los índices de columnas artificiales
    indices_a_eliminar = [i for i, nombre in enumerate(columnasLetras) if nombre.startswith('a')]

    # Eliminar columnas del tablero y nombres correspondientes
    for idx in sorted(indices_a_eliminar, reverse=True):
        tablero = np.delete(tablero, idx, axis=1)
        columnasLetras.pop(idx)

    print("VARIABLES ARTIFICIALES ELIMINADAS")
    ImprimirTabla(tablero, filasLetras, columnasLetras)

    return tablero

def reconstruirSimplex(tablero, c, obj, filasLetras, columnasLetras, procedimiento):

    zValor = 0

    if obj == "max":
        zValor = 1
    elif obj == "min":
        zValor = -1
        c = -1*c

    zFila = np.zeros((1, len(columnasLetras)), dtype=float)

    for i in range(np.shape(c)[1]):
            zFila[0][i+1] = c.flatten()[i]

    #agregar zfila al tablero
    print("zfila: ", zFila)
    tablero[0, 0] = zValor
    if procedimiento == "2fases":
        tablero[0, :] = -1* zFila
    else:
        tablero[0, :] = zFila
    #tablero[0, -1] = 0
    print("tablero antes de agregar la fila Z: ", tablero)

    return tablero

def hay_negativos_validos(filaZ, columnasLetras, fase, tablero):
    for i, valor in enumerate(filaZ):
        col_name = columnasLetras[i + 1]  # +1 por el desplazamiento por Z
        if valor < 0:
            if fase == "2fases" and col_name.startswith('a'):
                return True
            if fase in ("simplex", "mixto") and (col_name.startswith('x') or col_name.startswith('h')):
                return True
    return False

def Pivoteo2(tablero, filasLetras, columnasLetras):
    print("Iniciando Pivoteo2...")

    for i in range(1, len(filasLetras)):  # Saltar Z
        var_basica = filasLetras[i]

        if var_basica not in columnasLetras:
            print(f"Variable {var_basica} no está en columnas, se ignora.")
            continue

        fila_idx = i
        col_idx = columnasLetras.index(var_basica)

        pivote = tablero[fila_idx][col_idx]
        if abs(pivote) < 1e-8:
            print(f"La posición de pivote ({var_basica}, {var_basica}) es cero, no se puede normalizar.")
            continue

        tablero[fila_idx] = tablero[fila_idx] / pivote

        for j in range(len(tablero)):
            if j != fila_idx:
                factor = tablero[j][col_idx]
                tablero[j] = tablero[j] - factor * tablero[fila_idx]

        print(f"Pivoteo aplicado para variable básica '{var_basica}' en columna '{columnasLetras[col_idx]}'.")

    return tablero

def Pivoteo3(tablero, filasLetras, columnasLetras):
    print("Iniciando Pivoteo3...")

def Simplex(A, b, c, ci, signos, obj, filasLetras, columnasLetras):
    print("Simplex")
    
    procedimiento = ''
    aux = False

    menorIgual = 0
    mayorIgual = 0
    for i in range(len(signos)):
        if signos[i] == 1:
            menorIgual += 1
        elif signos[i] == 3:
            mayorIgual += 1

    if menorIgual > 0 and mayorIgual == 0:
        procedimiento = 'simplex'
        c = -1*c
    if menorIgual == 0 and mayorIgual > 0:
        procedimiento = '2fases'
    if menorIgual > 0 and mayorIgual > 0:
        procedimiento = "mixto"

    AA = FormaAmpliada(A, b, c, ci, signos, obj, procedimiento, filasLetras, columnasLetras)


    aux = True
    
    while aux: 
        #maximizacion menor igual
        zFuncion = AA[0, 1:-1]
        aux = hay_negativos_validos(zFuncion, columnasLetras, procedimiento, AA)

        if not aux and procedimiento == "2fases": #minimizacion mayor igual
            print("cambiando a simplex")
            AA = elimVarArtificiales(AA, filasLetras, columnasLetras)

            
            AA = reconstruirSimplex(AA, c, obj, filasLetras, columnasLetras,procedimiento)
            print("AA 2 fases: \n {}", AA)

            procedimiento = "simplex"

            AA = Pivoteo2(AA,filasLetras,columnasLetras)
            aux = False
            continue

        if not aux and procedimiento == "mixto" and obj == "min":
            print("cambiando a simplex mixto (min)")

            print("AA :", AA)
            AA = elimVarArtificiales(AA, filasLetras, columnasLetras)
            procedimiento = "simplex"
            
            print("AA :", AA)
            # Pasar c como está, sin modificarlo más
            #c = -1*c #si sale positivo no se borra, pero sino se borra
            AA = reconstruirSimplex(AA, c, "min", filasLetras, columnasLetras,procedimiento)
            print("AA: ", AA)
            AA = Pivoteo2(AA,filasLetras,columnasLetras)

            print(AA)
            aux = False
            continue

        if not aux and procedimiento == "mixto" and obj == "max":
            print("cambiando a simplex mixto (max)")

            AA = elimVarArtificiales(AA, filasLetras, columnasLetras)
            procedimiento = "simplex"

            #print("AA :", AA)
            # Pasar c como está, sin modificarlo más
            #c = -1*c #si sale positivo no se borra, pero sino se borra
            AA = reconstruirSimplex(AA, c, "min", filasLetras, columnasLetras,procedimiento)
            print("antes de pivoteo2: ", AA)
            AA = Pivoteo2(AA,filasLetras,columnasLetras)

            print(AA)
            aux = True
            continue
            # c = -1*c
            # AA = reconstruirSimplex(AA, c, "max", filasLetras, columnasLetras)
            # aux = True
            # continue

            #--------------------------------

        # if not aux and procedimiento == "mixto" and obj == "min":
             
        #     print("Entreo a aux normal, para mixto min")
        #     print(AA)
        #     AA = elimVarArtificiales(AA, filasLetras, columnasLetras)
        #     print(AA)
        #     #  procedimiento = "simplex"
        #     #  AA = reconstruirSimplex(AA, c, "min", filasLetras, columnasLetras)
        #     aux = False
        #     continue

        
        if aux:
            AA = Pivoteo(AA, filasLetras, columnasLetras)
    return AA

def ImprimirTabla(tablero, filasLetras, columnasLetras):
    # Crear tabla con encabezados y filas
    print("Dentro de imprimir tabla")
    print("filasLetras: ", filasLetras)
    print("columnasLetras: ", columnasLetras)
    #print("tablero: ", tablero)
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
    columnasLetras = []
    filasLetras = []

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

    print("Ingresa cada restricción, formato:\n2*x1 + 3*x2 <= 300\n")
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
    print("Ahora ingresa la función obj, formato:\n30000*x1 + 4000*x2\n")
    obj_str = input("> ")
    c = np.array(parse_obj(obj_str, v)).reshape(1, -1)
    print("Ahora el tipo de modo 'max' o 'min'\n")

    modo = input(">")

    print("Matriz A (coeficientes de restricciones):\n")
    print(A)
    print("Vector b (lado derecho de restricciones):\n")
    print(b)
    print("Signos de las restricciones:")
    print(signos)
    print("Vector c (coeficientes de la función obj):\n")
    print(c)
    return Simplex(A, b, c, ci, signos, modo, filasLetras, columnasLetras)

tablero = menu()

print(tablero)