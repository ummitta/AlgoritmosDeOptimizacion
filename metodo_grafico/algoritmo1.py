import numpy as np
from matplotlib import pyplot as plt
import random as ra
from scipy.spatial import ConvexHull
import sympy as sp

def FindVertors(A,b):

    V = np.empty((0,2))

    #print(V)

    for i in range(len(b)-1):
        for j in range(0,len(b)):

            if i != j:

                Ax = np.empty((2,2),dtype=int)
                bx = np.empty((2,1),dtype=int)

                Ax[0,0],Ax[0,1] = A[i][0],A[i][1]
                Ax[1,0],Ax[1,1] = A[j][0],A[j][1]

                bx[0,0] = b[i][0]
                bx[1,0] = b[j][0]

                det = np.linalg.det(Ax)

                if det != 0.0:
                    #print(det)
                    Vx,Vy = np.linalg.solve(Ax,bx)
       
                    vertex = np.array([Vx[0],Vy[0]])
                    #vertex = vertex.astype(int)
                    #print(vertex)
                    V = np.vstack((V,vertex))
                    
    V = V.astype(int)
    return V


def FeasibleVectors(V,A,b,sign):

    VF = np.empty((0,2))

    for i in range(len(V)):
        print("i: ",i)

        state = True

        for j in range(len(b)):
            print("j: ", j)

            print(f"Vertice: {V[i][0],V[i][1]}")

            if sign[j] == 1:
                print("<=")
                print(f"{A[j][0]} * {V[i][0]} + {A[j][1]} * {V[i][1]} <= {b[j]}")
                operation = A[j][0] * V[i][0] + A[j][1] * V[i][1] <= b[j]
                print(operation)

                if operation == False:
                    state = False


            elif sign[j] == 3:
                print(">=")
                print(f"{A[j][0]} * {V[i][0]} + {A[j][1]} * {V[i][1]} >= {b[j]}")
                operation = A[j][0] * V[i][0] + A[j][1] * V[i][1] >= b[j]
                print(operation)
                if operation == False:
                    state = False


        if state == True:
            row = np.array([V[i][0],V[i][1]])
            VF = np.vstack((VF,row))
    return VF


def Evaluate(VF,C):

    VE = np.empty((0,1),dtype=int)

    for i in range(len(VF)):
  

        aux = VF[i][0] * C[0] + VF[i][1] * C[1]

        row = np.array([aux])

        VE = np.vstack((VE,row))
    

    Ve = np.array(VE,dtype=int)
    return Ve

def Optime(objective,Ve,V):
    V = np.array(V,dtype=int)
    if objective == "maximizar":
        aux = Ve[0]
        index = 0
        for i in range(1,len(Ve)):
            
            if Ve[i] > aux:
                aux = Ve[i]
                index = i
    
    elif objective == "minimizar":
        aux = Ve[0]
        index = 0
        for i in range(1,len(Ve)):
            
            if Ve[i] < aux:
                aux = Ve[i]
                index = i


    optimizeVertex = V[index]


    return optimizeVertex,aux 

def evaluar_funciones(V, b):
    x1 = np.linspace(-100,100,100)
    return ((b - (V[0] * x1))/V[1]), f"({b} - ({V[0]} * x))/{V[1]}"

def graficar(sol, A, b, c, signo,VF):
    plt.grid()


    x = sol[0]
    y = sol[1]
    print('c: ',c)
    print('c[0]: ', c[0][0])
    print('c[1]: ', c[0][1])
    Z_eval = c[0][0] * x + c[0][1] * y 
    # graficar en consola el punto optimo

    print("El punto optimo esta en:")
    print("---------")
    print("| x =", int(x), '|')
    print("---------")
    print("| y =", int(y), "|")
    print("---------")
    print('al evaluar Z obtenemos:', Z_eval)

    plt.xlim(-2,50)
    plt.ylim(-2,50)
    plt.xticks(np.arange(-2,50,2))
    plt.yticks(np.arange(-2,50,2))
    #asdasd
    # ploteando eje x e y
    plt.axhline(0,color="black",linewidth=1)
    plt.axvline(0,color="black",linewidth=1)

    for i in range(len(A)):
        # para cada restricción se decide como plotearla
        colors = [(ra.random(),ra.random(),ra.random()) for a in A ]

        if A[i][0] == 0 or A[i][1] == 0:
            if A[i][0] == 0:
                yGrafico = int(b[i]/A[i][1])
                
                print("yGrafico: ", yGrafico)
                plt.axhline(y=b[i]/A[i][1], color=colors[i], label=f"y = {int(b[i]/A[i][1])}", linestyle="--")
            if A[i][1] == 0:
                plt.axvline(x=b[i]/A[i][0], color=colors[i], label=f"x = {int(b[i]/A[i][0])}", linestyle="--")
        else:
            if A[i][0] > 0 and A[i][1] > 0:
                e = evaluar_funciones(A[i], b[i])[0]
                plt.plot(np.linspace(-100,100,100), e, label=evaluar_funciones(A[i], b[i])[1], color="g", linestyle="--")

    hull = ConvexHull(VF)
    for simplex in hull.simplices:
        plt.plot(VF[simplex, 0], VF[simplex, 1], 'g--')
    plt.fill(VF[hull.vertices, 0], VF[hull.vertices, 1], 'lightgreen', alpha=0.3)

    # ploteando todos los factibles
    for v in VF:
        plt.plot(v[0],v[1], 'bo')
    

    # ploteando el punto optimo
    plt.plot(x,y, 'ro')
    plt.text(x+.1,y+.3,f"({int(x)},{int(y)}) es el punto optimo del modelo.")

    funcion_objetivo = f"Z = {c[0][0]} * {x} + {c[0][1]} * {y} = {c[0][0] * x + c[0][1] * y}"



    # Agregar a la leyenda
    plt.plot([], [], label=funcion_objetivo, color='black', linestyle='-')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Solución Optima")
    plt.legend()
    plt.show()


def ModeloProgramacionLinea(coeficientes,restricciones,coeficienteFuncionObjetivo,signos,modo):

    V = FindVertors(coeficientes,restricciones)
    VF = FeasibleVectors(V,coeficientes,restricciones,signos)
    Ve = Evaluate(VF,c)
    optimizeVertex,profit =Optime(modo,Ve,VF)
    print('VF: ', VF)
    print('Ve: ', Ve)


    graficar(optimizeVertex,coeficientes,restricciones,coeficienteFuncionObjetivo,signos,VF)



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



    # 3) Salida final
    print("\nMatriz A (coeficientes de restricciones):")
    print(A)
    print("\nVector b (lado derecho de restricciones):")
    print(b)
    print("\nSignos de las restricciones:")
    print(signos)
    print("\nVector c (coeficientes de la función objetivo):")
    print(c)

    ModeloProgramacionLinea(A,b,c,signos,modo)





A = np.array([[1,0],
              [0,2],
              [3,2],
              [1,0],
              [0,1]])


b = np.array([[4],
              [12],
              [18],
              [0],
              [0]])

c = np.array([[30000],[50000]])
#signos
# < 0
# <= 1
# > 2
# >= 3
# != 4
sign = np.array([[1],[1],[1],[3],[3]])

#modos
#maximizar
#minimizar


menu()

#ModeloProgramacionLinea(A,b,c,sign,'maximizar')



