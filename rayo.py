import numpy as np
import cv2
from control_pid import controlDrone

class trayectoria_ray():

    def __init__(self, dt, distancia_deseada=1.2):
        self.dt = dt
        self.distancia_deseada = distancia_deseada
        self.trayectoria_activa = False
        self.finalizado = False

        #============================== Control PID ==============================
        self.pid_adelante = controlDrone([50.0, 2.5, 0.5], 0.05)
        self.pid_lateral  = controlDrone([150.0, 1.0, 1.8], 0.02)
        self.pid_altura   = controlDrone([160.0, 4.0, 1.2], 0.02, max_delta=10)
        self.pid_yaw      = controlDrone([35.0, 1.8, 0.5], 0.05)

        self.Po = np.zeros(3)
        self.P_final = np.zeros(3)
        self.W = np.zeros(3)

        self.dist_ant = 0.0
        self.ux_ant = 0.0
        self.uy_ant = 0.0
        self.uz_ant = 0.0

    def iniciar_trayectoria(self, ux, uy, uz, norma, fueraZona, EnRuta):

        self.Po = np.array([-ux * norma, -uy * norma, -uz * norma])

        self.W = -self.Po
        norm_W = np.linalg.norm(self.W)
        if norm_W > 0:
            self.W = self.W / norm_W
        else:
            self.W = np.array([0.0, 0.0, -1.0])

        self.P_final = self.distancia_deseada * (-self.W)

        self.finalizado = False
        self.trayectoria_activa = True

        self.pid_adelante.reset()
        self.pid_altura.reset()
        self.pid_lateral.reset()
        self.pid_yaw.reset()

    def reiniciar(self):
        self.trayectoria_activa = False

    # -------------------------------------------------------------
    # Selección de candidato: de entre m1/m2/m3 (los que existan),
    # descarta los corruptos y elige los que estan mejor alineados
    #(en ruta)
    #--------------------------------------------------------------

    def calcular_velocidades(self, candidato, fueraZona, EnRuta):

        if candidato is None:
            return (0, 0, 0, 0), "Filtro: candidatos inválidos", self.finalizado

        ux_cam, uy_cam, uz_cam = candidato.ux, candidato.uy, candidato.uz
        norma = candidato.norma
        distancia = norma #solo para logica de frenado tiene magnitud de la norma
        yaw_visual = candidato.yaw

        P_actual = np.array([-ux_cam * norma, -uy_cam * norma, -uz_cam * norma])

        V = self.Po - P_actual
        norm_V = np.linalg.norm(V)

        # Se queda flotando donde está
        distancia_al_objetivo = distancia - self.distancia_deseada

        if distancia_al_objetivo <= 0.20:
            P_ideal = self.P_final.copy()
        else:
            delta_t = 0.4
            t_val = norm_V + delta_t
            P_ideal = self.Po + (t_val * self.W)

        self.finalizado = False

        if fueraZona:
            P_ideal = P_actual.copy()

        Error_body = P_ideal - P_actual

        Ex = Error_body[0]
        Ey = Error_body[1]
        Ez = Error_body[2]
        ETheta = yaw_visual

        if fueraZona:
            self.finalizado = False  # no evaluamos mientras el dron está flotando sin control
        elif distancia_al_objetivo <= 0.20:
            convergencia_lateral = abs(yaw_visual) <= 0.15
            self.finalizado = (abs(distancia_al_objetivo) <= 0.15) and convergencia_lateral
        else:
            self.finalizado = False

        v_lateral = self.pid_lateral.estimacion_vel_zm(Ex, self.dt)
        v_adelante = self.pid_adelante.estimacion_vel_zm(-Ez, self.dt)
        v_altura = self.pid_altura.estimacion_vel_zm(Ey, self.dt) 
        v_yaw = self.pid_yaw.estimacion_vel_zm(ETheta, self.dt)

        debug_msg = (f"U({ux_cam:.2f},{uy_cam:.2f},{uz_cam:.2f}) roll:{np.degrees(yaw_visual):+.1f}° | "
                     f"Err-> X:{Ex:.2f}m | Y:{Ey:.2f}m | Z:{Ez:.2f}m | fin: {self.finalizado}")

        if fueraZona:
            return (0, 0, 0, 0), debug_msg, self.finalizado
        else:
            return (v_lateral, v_adelante, v_altura, v_yaw), debug_msg, self.finalizado
