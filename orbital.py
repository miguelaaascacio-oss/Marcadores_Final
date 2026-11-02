import numpy as np
import math
from control_pid import controlDrone

class trayectoria_orb():

    def __init__(self, omega, sentido, grados, dt):
        self.omega = omega
        self.omega_inicial = omega
        self.sentido = sentido
        self.grados = grados
        self.dt = dt
        self.angulo_recorrido = 0.0
        self.trayectoria_activa = False
        self.finalizado = False

	#============================== Control PID ==============================

	# 1. Adelante/Atrás (Controla el eje Y del mundo)
        self.pid_radial = controlDrone([25.0, 8.0, 0.0], 0.02)  

	# 2. Izquierda/Derecha (Controla el eje X del mundo)
        self.pid_tangencial = controlDrone([24.0, 7.5, 0.0], 0.02) 

	# 3. Arriba/Abajo (Controla el eje Z del mundo)
        self.pid_altura   = controlDrone([16.0, 8.0, 0.0], 0.04) 

	# 4. Giro sobre su eje (Controla ETheta en radianes)
	# zm = 0.14 radianes equivale a ~8 grados
        self.pid_yaw      = controlDrone([95.0, 2.5, 1.2], 0.02)  
        
        #variables internas
        self.K_ff = -16.0 # Ajusta este valor experimentalmente
        self.angulo_objetivo_a_rotar = math.radians(self.grados)  
                
        self.rd = 0.0          # radio deseado (m)
        self.hd = 0.0          # altura deseada (m)
        self.theta0 = 0.0      # ángulo del ancla
        self.theta_d_rel = 0.0 # referencia relativa
        self.theta_rel = 0.0   # medido relativo

    def iniciar_trayectoria(self, norma, altura, theta, grados_orbitales=None):
        if grados_orbitales is not None:
            self.grados = float(grados_orbitales)
            self.angulo_objetivo = math.radians(abs(self.grados))
            self.sentido = -1 if self.grados < 0 else 1

        self.hd = altura
        self.rd = math.sqrt(max(norma**2 - altura**2, 0.0))
        self.theta0 = theta
        self.theta_d_rel = 0.0
        self.theta_rel = 0.0
        self.angulo_recorrido = 0.0

        self.omega = self.omega_inicial
        self.finalizado = False
        self.trayectoria_activa = True

        self.pid_radial.reset()
        self.pid_tangencial.reset()
        self.pid_altura.reset()
        self.pid_yaw.reset()

    def reiniciar(self):
        self.trayectoria_activa = False
        self.finalizado = False

    def calcular_velocidades(self, norma, altura, yaw_visual, theta):
        # --- estado actual ---
        r_actual = math.sqrt(max(norma**2 - altura**2, 0.0))
        h_actual = altura
        d = self.theta0 - theta
        self.theta_rel = math.atan2(math.sin(d), math.cos(d))

        # --- referencia: avanza hasta el ángulo objetivo ---
        if abs(self.theta_d_rel) < self.angulo_objetivo:
            self.theta_d_rel += self.omega * self.dt * self.sentido

        # --- errores ---
        Er     = r_actual - self.rd
        Eh     = self.hd - h_actual
        Etheta = self.theta_rel - self.theta_d_rel 

        # --- fin: ángulo MEDIDO en el sentido de la órbita ---
        self.angulo_recorrido = self.theta_rel * self.sentido
        if self.angulo_recorrido >= self.angulo_objetivo:
            self.finalizado = True
            progreso = f"COMPLETADO: {math.degrees(self.angulo_recorrido):.1f}° (objetivo {self.grados:.1f}°)"
            return (0, 0, 0, 0), f"[{progreso}] Manteniendo posición estática...", True
                
        # --- PID ---
        v_adelante = int(self.pid_radial.estimacion_vel_zm(Er, self.dt))
        v_altura   = int(self.pid_altura.estimacion_vel_zm(Eh, self.dt))
        v_yaw      = int(self.pid_yaw.estimacion_vel_zm(yaw_visual, self.dt))
        v_roll_pid = int(self.pid_tangencial.estimacion_vel_zm(Etheta, self.dt))

        v_ff = int(r_actual * self.omega * self.sentido * self.K_ff)  
        v_lateral = v_ff + v_roll_pid

        progreso = f"{math.degrees(self.angulo_recorrido):.1f}°/{math.degrees(self.angulo_objetivo):.0f}°"
        msg = (f"[Progreso: {progreso}] Er={Er:.3f} | Eh={Eh:.3f} | "
               f"Etheta={Etheta:.3f} actual: {math.degrees(self.theta_rel):.1f}° | "
               f"lat:{v_lateral} fb:{v_adelante} ud:{v_altura} yaw:{v_yaw}")
        return (v_lateral, v_adelante, v_altura, v_yaw), msg, False
