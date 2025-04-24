import numpy as np
from matplotlib import pyplot as plt
import random as ra
from scipy.spatial import ConvexHull

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

print(len(b))

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

V = FindVertors(A,b)

print(len(V))
print(len(b))
VF = FeasibleVectors(V,A,b,sign)
print("factibles: ",VF)

Ve = Evaluate(VF,c)
print(Ve)

optimizeVertex,profit =Optime('maximizar',Ve,VF)

print(optimizeVertex,profit)



print(optimizeVertex)

def evaluar_funciones(V, b):
    x1 = np.linspace(-100,100,100)
    return ((b - (V[0] * x1))/V[1]), f"({b} - ({V[0]} * x))/{V[1]}"

def graficar(sol, A, b, c, signo):
    plt.grid()
    V_np = np.array(FindVertors(A, b), dtype=int)
    A_np = np.array(A)
    signo_np = np.array(signo)
    b_np = np.array(b)
    c_np = np.array(c)

    x = sol[0]
    y = sol[1]
    Z_eval = c[0] * x + c[1] * y 
    # graficar en consola el punto optimo
    print("El punto optimo esta en:")
    print("---------")
    print("| x =", int(x), "|")
    print("---------")
    print("| y =", int(y), "|")
    print("---------")
    print('al evaluar Z obtenemos:', Z_eval)

    plt.xlim(-2,50)
    plt.ylim(-2,50)
    plt.xticks(np.arange(-2,100,5))
    plt.yticks(np.arange(-2,100,5))
    #asdasd
    # ploteando eje x e y
    plt.axhline(0,color="black",linewidth=1)
    plt.axvline(0,color="black",linewidth=1)

    for i in range(len(A)):
        # para cada restricción se decide como plotearla
        colors = [(ra.random(),ra.random(),ra.random()) for a in A ]

        if A[i][0] == 0 or A[i][1] == 0:
            if A[i][0] == 0:
                plt.axhline(y=b[i]/A[i][1], color=colors[i], label=f"y = {b[i]/A[i][1]}", linestyle="--")
            if A[i][1] == 0:
                plt.axvline(x=b[i]/A[i][0], color=colors[i], label=f"x = {b[i]/A[i][0]}", linestyle="--")
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

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Solución Optima")
    plt.legend()
    plt.show()


graficar(optimizeVertex,A,b,c,sign)