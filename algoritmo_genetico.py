# Algoritmo Genético para la Optimización de Funciones x

import random as ra
import numpy as np
import sympy as sp
import json
import matplotlib.pyplot as plt

with open('.\Anexos\Data.json', 'r') as file:
    data = json.load(file)

print(data)