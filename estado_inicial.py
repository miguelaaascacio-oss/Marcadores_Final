import time
import sys
import select  
import math
    
class ValidadorEstadoInicial(): 
    def __init__(self, sock, funcion_leer_datos, instancia_tello=None, callback_emergencia=None):
        """
        Clase encargada de buscar un ancla estable filtrando el ruido del sensor.
        
        :param sock: El socket UDP abierto.
        :param funcion_leer_datos: La función 'leer_datos' externa.
        :param instancia_tello: (Opcional) Objeto Tello para enviar comandos de parada.
        :param callback_emergencia: (Opcional) Función externa a ejecutar en caso de emergencia.
        """
        self.sock = sock
        self.leer_datos = funcion_leer_datos
        self.tello = instancia_tello
        self.emergencia_cb = callback_emergencia
        
        
    def _is_key_pressed(self):
        """Método interno para verificar el teclado."""
        return select.select([sys.stdin], [], [], 0.001)[0]
   
    def obtener_estado_estable(self, timeout_local=10, tolerancia=0.15):
        """
        Lee datos hasta conseguir un consenso estable.
        Requiere que 2 de 3 lecturas consecutivas no difieran más de la tolerancia del ancla.
        Retorna: (ux_cam, uy_cam, uz_cam, norma, yaw_visual)
        """
        referencia_inicial = None
        intentos = 0
        lecturas_validas = 0
        ultimo_valido = None
        
        ultimo_comando = time.time()
        tiempo_inicio_local = time.perf_counter()

        while True:
            # 1. Temporizador de seguridad
            if time.perf_counter() - tiempo_inicio_local >= timeout_local:
                print(f"\n¡No se logró estabilizar el marcador en {timeout_local} segundos!")
                raise TimeoutError("Timeout buscando estabilidad en el marcador")

            # 2. Control de interrupción por teclado (Emergencia)
            if self._is_key_pressed():
                key = sys.stdin.read(1)
                if key and self.emergencia_cb:
                    self.emergencia_cb() # Ejecuta la función de emergencia externa
                    raise Exception("Emergencia activada durante la inicialización")
                    
            # 3. Mantener al dron en hover
            if time.time() - ultimo_comando > 0.5:
                if self.tello:
                    self.tello.send_rc_control(0, 0, 0, 0)
                ultimo_comando = time.time()
                    
            # 4. Lectura de datos usando la función inyectada
            m1, m2, m3, fueraZona, EnRuta = self.leer_datos(self.sock)
            
            if m1 is None or not EnRuta:
                time.sleep(0.05)
                continue
                
            candidatos = [c for c in (m1, m2, m3) if c is not None]
            
            mejor_candidato = None
            for c in candidatos:
                if c.norma < 0:
                    continue

                entradas = [c.ux, c.uy, c.uz, c.norma, c.yaw, c.altura, c.theta]
                if any(math.isnan(v) or math.isinf(v) for v in entradas):
                    continue  # candidato corrupto

                if mejor_candidato is None or abs(c.yaw) < abs(mejor_candidato.yaw):
                    mejor_candidato = c

            if mejor_candidato is None:
                # Ningún candidato de este frame cae dentro del margen esperado
                referencia_inicial = None
                print("Sin candidato válido dentro del margen de distancia. Reintentando...")
                time.sleep(0.05)
                continue

            medicion_actual = (
                mejor_candidato.ux, mejor_candidato.uy, mejor_candidato.uz,
                mejor_candidato.norma, mejor_candidato.yaw, mejor_candidato.altura, mejor_candidato.theta,
                fueraZona, EnRuta
            )

            # Fijar el ancla inicial si está vacía
            if referencia_inicial is None:
                referencia_inicial = medicion_actual
                intentos = 0
                lecturas_validas = 0
                ultimo_valido = medicion_actual
                time.sleep(0.05)
                continue

            intentos += 1

            # Comprobar margen de tolerancia
            dentro_del_margen = True
            # Índices: 0:ux, 1:uy, 2:uz, 3:norma, 4:yaw_visual
            for i in range(5):
                a = medicion_actual[i]
                r = referencia_inicial[i]

                if i == 3:  # norma
                    if r == 0:
                        if abs(a) > 0.15:
                            dentro_del_margen = False
                            break
                    elif abs(a - r) > abs(r) * tolerancia:
                        dentro_del_margen = False
                        break
                else:  # Vectores y Ángulos
                    if abs(a - r) > 0.20:
                        dentro_del_margen = False
                        break

            if dentro_del_margen:
                lecturas_validas += 1
                ultimo_valido = medicion_actual

            # Decisión probabilística tras 3 intentos (Consenso 2/3)
            if intentos == 3:
                if lecturas_validas >= 2:
                    print(f"Estado inicial validado (Consenso: {lecturas_validas}/3).")
                    return ultimo_valido
                else:
                    print(f"Ruido detectado (Consenso: {lecturas_validas}/3). Reiniciando ancla...")
                    referencia_inicial = None

            time.sleep(0.05)
