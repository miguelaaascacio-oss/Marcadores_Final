import math   

class trayectoria_rot():
    def __init__(self, dt):
        self.dt = dt
        self.trayectoria_activa = False
        self.finalizado = False

        self.angulo_objetivo = 0.0
        self.grados_recorridos = 0.0
        self.velocidad_angular_estimada = 0.0

        self.COMANDO_YAW = 30
        self.W_MAX = 35.0        # °/s a ese comando
        self.ALFA = 0.20
        self.FRAC_MAX = 2.0      
        
        self.MARGEN_CENTRADO = math.radians(3.0) 

    def iniciar_trayectoria(self, angulo_requerido):
        self.angulo_objetivo = float(angulo_requerido)
        self.finalizado = False
        self.trayectoria_activa = True
        self.grados_recorridos = 0.0
        self.velocidad_angular_estimada = 0.0
        print(f"¡Iniciando giro dirigido de {self.angulo_objetivo:.2f}°!")

    def reiniciar(self):
        self.trayectoria_activa = False
        self.finalizado = False
        self.grados_recorridos = 0.0
        self.velocidad_angular_estimada = 0.0

    def calcular_velocidades(self, marcador_en_ruta, yaw_visual=None):
        if abs(self.angulo_objetivo) < 1.0:
            self.finalizado = True
            return (0, 0, 0, 0), "Giro omitido (ángulo cercano a 0°).", True

        sentido = 1 if self.angulo_objetivo < 0 else -1
        angulo_abs = abs(self.angulo_objetivo)

        # detener al detectar un marcador en ruta
        if marcador_en_ruta:
            if yaw_visual is None or abs(yaw_visual) <= self.MARGEN_CENTRADO:
                self.finalizado = True
                ystr = "N/A" if yaw_visual is None else f"{math.degrees(yaw_visual):+.1f}°"
                return (0, 0, 0, 0), f"Giro completado: marcador centrado (yaw={ystr}).", True

        #estimación de lo girado
        self.velocidad_angular_estimada += self.ALFA * (self.W_MAX - self.velocidad_angular_estimada)
        self.grados_recorridos += self.velocidad_angular_estimada * self.dt

        if self.grados_recorridos >= self.FRAC_MAX * angulo_abs:
            self.finalizado = True
            msg = (f"sin detectar marcador en ruta. Ángulo objetivo: {self.angulo_objetivo:.2f}°")
            return (0, 0, 0, 0), msg, True

        ystr = "N/A" if yaw_visual is None else f"{math.degrees(yaw_visual):+.1f}°"
        msg = f"Girando... est:{self.grados_recorridos:.1f}° / obj:{angulo_abs:.1f}° (límite {self.FRAC_MAX * angulo_abs:.1f}°)"
        return (0, 0, 0, int(self.COMANDO_YAW * sentido)), msg, False
