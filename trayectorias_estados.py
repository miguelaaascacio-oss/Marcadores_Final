from djitellopy import Tello
from control_pid import controlDrone
from orbital import trayectoria_orb
from rayo import trayectoria_ray
from rotacion import trayectoria_rot
from estado_inicial import ValidadorEstadoInicial
from enum import Enum
from collections import namedtuple

import socket
import struct
import time
import sys
import select
import tty
import termios
import math
import numpy as np

# ====================== CARGAR MAPA =============================
Trayectoria = np.load("Trayectoria.npy")
print(Trayectoria)
indice_segmento = 0
ultimo_paquete = 0.0
seg_actual = Trayectoria[indice_segmento]
print(f"Misión cargada. Total de puntos a seguir: {len(Trayectoria)}")
print(f"Segmento inicial -> ID Objetivo: {seg_actual['ID']} | Distancia tramo: {seg_actual['distancia']:.2f}m")

# ====================== CONFIGURACIÓN TERMINAL Y COMUNICACIÓN ======================
old_settings = None
DatosMarcador = namedtuple('DatosMarcador', ['ux','uy','uz','norma','yaw', 'altura', 'theta'])

def restaurar_terminal():
    global old_settings
    if old_settings is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nTerminal restaurada.")

def iniciar_socket():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    print(f"Escuchando paquetes UDP en {UDP_IP}:{UDP_PORT}...")
    return sock

def leer_datos(sock):
    global ultimo_paquete
    data = None
    try:
        while True:
            data, addr = sock.recvfrom(1024)
    except BlockingIOError:
        pass

    if data and len(data) == 170:
        ultimo_paquete = time.time()
        vals = struct.unpack('<21d??', data)
        m1 = DatosMarcador(*vals[0:7])

        if m1.norma < 0:                     # nada detectado -> todo None (misma lógica de siempre)
            return None, None, None, None, None

        m2_raw = DatosMarcador(*vals[7:14])
        m3_raw = DatosMarcador(*vals[14:21])
        m2 = m2_raw if m2_raw.norma >= 0 else None
        m3 = m3_raw if m3_raw.norma >= 0 else None
        fueraZona, EnRuta = vals[21], vals[22]

        return m1, m2, m3, fueraZona, EnRuta

    return None, None, None, None, None
    
def activar_emergencia():
    global corriendo, emergencia
    print("EMERGENCIA ACTIVADA!")
    emergencia = True
    corriendo = False
    tello.send_rc_control(0, 0, 0, 0)
    try:
        tello.emergency()
    except:
        pass
    restaurar_terminal()  
    
def seleccionar_marcador(m1, m2, m3, yaw_ref):
    candidatos = [c for c in (m1, m2, m3)
                  if c is not None and 0 <= c.norma < 900
                  and not math.isnan(c.yaw)]
    if not candidatos:
        return None
    if len(candidatos) >= 2 or yaw_ref is None:
        return min(candidatos, key=lambda c: abs(c.yaw))
    c = candidatos[0]
    if abs(c.yaw - yaw_ref) > UMBRAL_DELTA_YAW:
        return None                      
    return c
    
def is_key_pressed():
    return select.select([sys.stdin], [], [], 0.001)[0]
    
# ====================== INICIALIZACIÓN DEL TELLO ======================
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")

tello.streamoff()
time.sleep(1)
tello.streamon()
time.sleep(2)

sock = iniciar_socket()

# ============================== VARIABLES GLOBALES ==============================
corriendo = True
emergencia = False
mision_completa = False
dt = 1.0/20.0        # 20Hz
frames_perdidos = 15							

# ======================== MAQUINA DE ESTADOS ==========================

class EstadoDron(Enum):
    REPOSO = 1
    PERDER = 2
    ELEVANDO = 3
    TRAYECTORIA = 4
    RAYO = 5
    GIRO = 6
    ORBITAL = 7
    FIN = 8
# ======================= INSTANCIACIÓN DE CLASES =======================

# 1. Instanciamos el Validador de Estado Inicial
validador = ValidadorEstadoInicial(
    sock=sock, 
    funcion_leer_datos=leer_datos, 
    instancia_tello=tello, 
    callback_emergencia=activar_emergencia
)

# 2. Instanciamos el controlador que queramos usar en esta prueba (Ej: Orbital)
controlador_orb = trayectoria_orb(omega=0.2, sentido=-1, grados=50.0, dt=dt)
controlador_rayo = trayectoria_ray(dt=dt, distancia_deseada=1.2)
controlador_rot = trayectoria_rot(dt=dt)

# ========================== ESPERA DE VIDEO ==========================
while time.time() - ultimo_paquete > 0.5:
    leer_datos(sock)
    time.sleep(0.05)
print("Video listo")

old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno()) 
    
print("\n=== SISTEMA LISTO ===")
print("Presiona ENTER para definir el umbral y comenzar el vuelo...")
input("→ Presiona ENTER cuando estés listo: ")
print("Umbral definido - Despegando en 3 segundos...\n")
time.sleep(3)

# ======================== DESPEGUE =======================================
print("Despegando...")
tello.takeoff()
time.sleep(1) 

inicio = time.time()
while time.time() - inicio < 2.5:
    ciclo_inicio = time.time()
    tello.send_rc_control(0, 0, 0, 0)			#inicia muy arriba
    elapsed = time.time() - ciclo_inicio
    time.sleep(max(0, dt - elapsed))

print("¡Iniciando vuelo autónomo de 30 segundos!")
tiempo_inicio = time.perf_counter()

# ============================== BUCLE PRINCIPAL ==============================

estado_actual = EstadoDron.REPOSO
estado_anterior = None  
tiempo_inicio_reposo = time.perf_counter()
B_GIRO = True
B_TRAYECTORIA_LISTA = False
B_MARCADOR = False
B_DIST_MIN = False
B_ANGULO_REC = False
UMBRAL_DELTA_YAW = 0.2   # rad
yaw_ref = None


#Banderas elevando tiene su pd
B_ALTURA_OK = False
inicio_elevando = 0.0
frames_alt_ok = 0
ALTURA_MIN = 0.50       # altura mínima donde se da por válido 
FRAMES_ALT_OK = 1       # frames seguidos dentro del umbral
UMBRAL_ALT_M = 0.25     # m: error vertical aceptable (zona muerta de entrada)
BANDA_ALT_M = 0.10      # m: histéresis; una vez dentro, solo sales si pasas de 0.30 m
K_ALT = 50              # comando por metro de error
V_ALT_MIN, V_ALT_MAX = 20, 25
V_SUBIDA = 25           # subida cuando no se ve el marcador
T_MAX_ELEVANDO = 8.0   # seguridad
T_PERDER_CORTO = 4.5    
T_PERDER_LARGO = 6.5

D=Trayectoria[0]
inicio_ascenso = 0

try:
    while corriendo:
        tiempo_actual = time.perf_counter()
        ciclo_inicio = time.perf_counter()
        seg_actual = Trayectoria[indice_segmento]
        
        if estado_actual != estado_anterior:
            print(f"\n >>> CAMBIO DE ESTADO: [{estado_anterior.name if estado_anterior else 'INICIO'}] -> [{estado_actual.name}] <<<\n")
            print(f"Marcador en frente: {seg_actual['ID']}")
            print(f"Distancia siguiente: {seg_actual['distancia']:.2f}cm")
            print(f"Angulo siguiente: {seg_actual['angulo']:.2f}grados")
            if estado_actual == EstadoDron.REPOSO:
                tiempo_inicio_reposo = time.perf_counter()
            estado_anterior = estado_actual
            if estado_actual not in (EstadoDron.TRAYECTORIA, EstadoDron.RAYO, EstadoDron.ORBITAL): 
                yaw_ref = None
    
        # 1. BOTÓN DE EMERGENCIA GLOBLAL
        if is_key_pressed():
            key = sys.stdin.read(1)
            if key:   
                activar_emergencia()
                estado_actual = EstadoDron.FIN
                break
        
        if (tiempo_actual - tiempo_inicio) >= 350:
            print("¡Tiempo límite alcanzado! Finalizando prueba...")
            estado_actual = EstadoDron.FIN
            break 

        if emergencia:
            activar_emergencia()
            estado_actual = EstadoDron.FIN
            break
                    
        # 2. LEER DATOS SENSOR
        m1, m2, m3, fueraZona, EnRuta = leer_datos(sock)
        elegido = seleccionar_marcador(m1, m2, m3, yaw_ref)
        if elegido is not None:
            ux_cam, uy_cam, uz_cam, norma, yaw_visual, altura, theta = elegido
            if EnRuta and estado_actual in (EstadoDron.RAYO, EstadoDron.ORBITAL):
                yaw_ref = yaw_visual
            profundidad = uz_cam*norma*-1
            print(f"Profundidad:{profundidad:.2f}m | Yaw:{yaw_visual:.2f} | EnRuta:{EnRuta} | FZ:{fueraZona}")
        else:
            ux_cam = uy_cam = uz_cam = norma = yaw_visual = altura = theta = None
            if m1 is not None:
                print("Frame ignorado: marcador fuera de ruta")
        
        # 3. LÓGICA DE CONTROL BANDERAS MAQUINA ESTADOS
        if ux_cam is None or norma == -1 or not EnRuta:
            frames_perdidos += 1
        else:
            frames_perdidos = 0
            
        B_MARCADOR = (frames_perdidos < 15)
        
        
        #EJECUCION MAQUINA DE ESTADOS 
            
        if estado_actual == EstadoDron.REPOSO:
            print(f"[{estado_actual.name}] Esperando objetivo...")
            tello.send_rc_control(0, 0, 5, 0)
            B_TRAYECTORIA_LISTA = False	
            B_ALTURA_OK = False
            
            if B_MARCADOR:
                tello.send_rc_control(0, 0, 0, 0)
                estado_actual = EstadoDron.TRAYECTORIA
            else:
                # Acción de Reposo: Mantener posición
                tello.send_rc_control(0, 0, 0, 0)
                tiempo_espera = tiempo_actual - tiempo_inicio_reposo
                if tiempo_espera >= 15.0:
                    print("\n¡No se  detectan Marcadores! Aterrizando")
                    estado_actual = EstadoDron.FIN
                else:
                    if int(tiempo_espera * 20) % 20 == 0:
                        print(f"[{estado_actual.name}] Esperando objetivo...")
                             
        elif estado_actual == EstadoDron.TRAYECTORIA:
            if not B_MARCADOR:
                estado_actual = EstadoDron.REPOSO # Failsafe
                B_TRAYECTORIA_LISTA = False
                
            elif ux_cam is None or norma is None or norma == -1 or not EnRuta:
                # Frame hueco temporal: no anclar ni transicionar, solo esperar
                tello.send_rc_control(0, 0, 0, 0)
                
            elif not B_ALTURA_OK:
                tello.send_rc_control(0, 0, 0, 0)
                inicio_elevando = time.time()
                frames_alt_ok = 0
                estado_actual = EstadoDron.ELEVANDO
                
            else:
                if not B_TRAYECTORIA_LISTA:
                    print("Fijando ancla estable...")
                    estado_raw = validador.obtener_estado_estable(timeout_local=15)
                    
                    if estado_raw is None:
                        ux_raw, uy_raw, uz_raw, norma_ini, yaw_raw, altura_raw, theta_raw, fuera_raw, Ruta_raw = [None]*9
                    else:
                        ux_raw, uy_raw, uz_raw, norma_ini, yaw_raw, altura_raw, theta_raw, fuera_raw, Ruta_raw = estado_raw
                    
                    # Tu validación estricta
                    ancla_ok = (
                        ux_raw is not None 
                        and norma_ini is not None and norma_ini != -1
                        and Ruta_raw is True
                    )
                    
                    if not ancla_ok:
                        print("Fallo al fijar trayectoria. Regresando a Reposo...")
                        B_TRAYECTORIA_LISTA = False
                        B_MARCADOR = False
                        estado_actual = EstadoDron.REPOSO
                        continue
                        
                    # Inicialización usando solo las variables validadas
                    yaw_ref = yaw_raw
                    controlador_rayo.iniciar_trayectoria(ux_raw, uy_raw, uz_raw, norma_ini, fuera_raw, Ruta_raw)
                    controlador_orb.iniciar_trayectoria(norma_ini, altura_raw, theta_raw, grados_orbitales=float(Trayectoria[-1]['angulo']))
                    B_TRAYECTORIA_LISTA = True
                    
                # Una vez lista la trayectoria, decidimos si saltamos a Rayo u Orbital
                if B_TRAYECTORIA_LISTA:
                    if norma is None or norma == -1 or not EnRuta:
                        # Si justo este frame se perdió, esperamos 50ms al siguiente
                        pass
                    elif indice_segmento >= len(Trayectoria) - 1:
                    # Bypass: Ya estamos a la distancia correcta
                        print("¡Último segmento alcanzado! Cambiando a ORBITAL.")
                        estado_actual = EstadoDron.ORBITAL
                    elif B_GIRO:
                        tello.send_rc_control(0, 30, 5, 0) #Romper inercia
                        estado_actual = EstadoDron.RAYO
                        
        elif estado_actual == EstadoDron.RAYO:
            if not B_MARCADOR:
                print("Marcador perdido en Rayo (10 frames). Abortando...")
                estado_actual = EstadoDron.REPOSO # Failsafe
                B_TRAYECTORIA_LISTA = False
                controlador_rayo.reiniciar()
                
            elif ux_cam is None or norma == -1 or not EnRuta:
                # Pérdida temporal (< 10 frames): se ignora el frame y se mantiene el último estado
                pass
                
            else:
                resultado, msg, B_DIST_MIN = controlador_rayo.calcular_velocidades(elegido, fueraZona, EnRuta)
                
                if B_DIST_MIN:
                    tello.send_rc_control(0, 0, 0, 0)  #Romper inercia
                    time.sleep(0.2)
                    controlador_rayo.reiniciar()

                    indice_segmento += 1
                    seg_actual = Trayectoria[indice_segmento]
                    B_TRAYECTORIA_LISTA = False        # obliga a re-anclar después

                    if indice_segmento >= len(Trayectoria) - 1:
                        # siguiente = órbita: sin PERDER ni GIRO, va directo a ELEVANDO
                        B_GIRO = True
                        B_ALTURA_OK = False
                        inicio_elevando = time.time()  # lo que normalmente hace TRAYECTORIA
                        frames_alt_ok = 0
                        estado_actual = EstadoDron.ELEVANDO
                    else:
                        B_GIRO = False
                        inicio_ascenso = time.time()
                        estado_actual = EstadoDron.PERDER
                        
                elif resultado is not None:
                    tello.send_rc_control(*resultado)
                    print(msg) 
                else:
                    print(msg)
                    
                    
        elif estado_actual == EstadoDron.ELEVANDO:
            distancia_sig = seg_actual['distancia']
            es_orbita = indice_segmento >= len(Trayectoria) - 1
            subir_a_ciegas = (distancia_sig > 150) and not es_orbita
            visible = (uy_cam is not None) and bool(EnRuta)
            
            err_v = uy_cam * norma if visible else 0.0
            
            en_umbral = visible and (subir_a_ciegas or abs(err_v) <= UMBRAL_ALT_M)
            en_banda = (visible and frames_alt_ok > 0
                        and abs(err_v) <= UMBRAL_ALT_M + BANDA_ALT_M)
            en_piso = (visible and profundidad <= 2.0
                       and altura is not None and not math.isnan(altura)
                       and altura <= ALTURA_MIN and uy_cam < 0)

            if time.time() - inicio_elevando > T_MAX_ELEVANDO:
                print("ELEVANDO: tiempo máximo excedido. Abortando...")
                tello.send_rc_control(0, 0, 0, 0)
                estado_actual = EstadoDron.REPOSO

            elif not visible:
                if B_MARCADOR:
                    pass    #pasa hasta que se reinicie B_MARCADOR (perder 15 frames)
                else:
                    frames_alt_ok = 0
                    if subir_a_ciegas:
                        tello.send_rc_control(0, 0, V_SUBIDA, 0)
                    else:
                        tello.send_rc_control(0, 0, 0, 0)
                        estado_actual = EstadoDron.REPOSO

            elif en_umbral or en_piso or en_banda:
                tello.send_rc_control(0, 0, 0, 0)
                frames_alt_ok += 1
                if frames_alt_ok >= FRAMES_ALT_OK:
                    print(f"Altura correcta (Error altura={err_v:+.3f}). Pasando a anclar.")
                    B_ALTURA_OK = True
                    B_TRAYECTORIA_LISTA = False
                    estado_actual = EstadoDron.TRAYECTORIA
                    

            else:
                frames_alt_ok = 0
                v = K_ALT * err_v
                v = int(math.copysign(min(max(abs(v), V_ALT_MIN), V_ALT_MAX), v))
                tello.send_rc_control(0, 0, v, 0)
                print(f"[ELEVANDO] err={err_v:+.2f} m altura={altura:.2f} -> v_alt={v}")	
                
        elif estado_actual == EstadoDron.PERDER:
            t_subida = T_PERDER_LARGO if seg_actual['distancia'] > 150 else T_PERDER_CORTO

            tello.send_rc_control(0, 0, 25, 0)

            if time.time() - inicio_ascenso >= t_subida:
                tello.send_rc_control(0, 0, 0, 0)
                print("Marcador perdido de vista")
                controlador_rot.iniciar_trayectoria(seg_actual['angulo'])
                estado_actual = EstadoDron.GIRO
  
        
        elif estado_actual == EstadoDron.GIRO:
            marcador_pare = (elegido is not None) and bool(EnRuta)
            yaw_vis = elegido.yaw if elegido is not None else None  
            resultado, debug_msg, B_GIRO = controlador_rot.calcular_velocidades(marcador_pare, yaw_vis)
            tello.send_rc_control(*resultado)
            print(debug_msg)  
            
            if B_GIRO:
                print(f"Giro del segmento {indice_segmento} terminado. Frenando inercia...")
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)  # Breve pausa para estabilizar el Tello tras el giro
                controlador_rot.reiniciar()
                estado_actual = EstadoDron.REPOSO

        elif estado_actual == EstadoDron.ORBITAL:
            if not B_MARCADOR:
                print("Marcador perdido en Orbita (10 frames). Abortando...")
                estado_actual = EstadoDron.REPOSO # Failsafe
                B_TRAYECTORIA_LISTA = False
                controlador_orb.reiniciar()
                
            elif ux_cam is None or norma == -1:
                # Pérdida temporal (< 10 frames): se ignora el frame y se mantiene el último estado
                pass
                
            else:
                resultado, msg, B_ANGULO_REC = controlador_orb.calcular_velocidades(norma, altura, yaw_visual, theta)
                if B_ANGULO_REC:
                    print(f"Órbita completada: {msg} -> Misión terminada.")
                    tello.send_rc_control(0, 0, 0, 0)
                    estado_actual = EstadoDron.FIN
                elif resultado is not None:
                    tello.send_rc_control(*resultado)
                else:
                    print(msg)


        elif estado_actual == EstadoDron.FIN:
            # Acción final de seguridad
            tello.send_rc_control(0, 0, 0, 0)
            break # Salimos del bucle while para iniciar el aterrizaje

        # 5. CONTROL DE TIEMPO (20Hz) (Mantenido igual)
        elapsed = time.perf_counter() - ciclo_inicio
        time.sleep(max(0, dt - elapsed))
except Exception as e:
    print(f"Error principal: {e}")

finally:
    print("\nIniciando aterrizaje final...")
    tello.send_rc_control(0, 0, 0, 0)
    try: tello.land()
    except: pass
    try: sock.close()
    except: pass
    try: tello.streamoff()
    except: pass							
    try: tello.end()
    except: pass
    
    restaurar_terminal()
    print("Prueba completada y conexión cerrada.")
