import os
import numpy as np

K = np.array([[595.0, 0.0, -311.0],
              [0.0, 599.0, -237.0],
              [0.0, 0.0, -1.0]])
              
Marcador = np.array([[ 5.0,  5.0,  0],
                   [-5.0,  5.0,  0],
                   [-1.47, 2.80, 0],
                   [ 1.47, 2.80, 0],
                   [-5.0, -5.0,  0],
                   [ 5.0, -5.0,  0]])
                   
directorio = (
    "/home/armas/Documents/TesisMiguelArmas/Tesis_corregida/Prueba_marcador_Original/R_guardadas/"
)
archivos = [f for f in os.listdir(directorio) if f.endswith(".txt")]

archivos_ordenados = sorted(archivos, key=lambda x: int(x.replace(".txt", "")))

for archivo in archivos_ordenados:
  print("frame", archivo.replace(".txt", ""))
  ruta = os.path.join(directorio, archivo)

  with open(ruta, "r") as f:
    datos = np.array(f.read().split(), dtype=float).reshape(-1, 25)

  for d in datos:
    R = d[1:10].reshape(3, 3)
    t = d[10:13]
    leidos = d[13:25].reshape(6, 2)

  producto = R @ R.T

  p = K @ (R @ Marcador.T + t.reshape(3, 1))  # K (R X + t)
  proyectados = (p[:2] / p[2]).T  # (u, v) = (x/f, y/f)

  print("etiqueta", int(d[0]))
  print(producto)
  print(np.hstack([leidos, proyectados]))

