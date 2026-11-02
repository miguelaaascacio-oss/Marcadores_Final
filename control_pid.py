import numpy as np
class controlDrone():
    def __init__(self, k, zm, alpha_d=0.3, max_delta=8):
        self.k = k
        self.zm = zm
        self.e_ant = 0.0
        self.integral = 0.0
        self.e_ant = None
        self.alpha_d = alpha_d 
        self.derivativo_filtrado = 0.0
        self.max_delta = max_delta          
        self.v_ant = 0
        
    def reset(self):
        """Limpia la memoria del controlador para evitar el Derivative Kick"""
        self.e_ant = None
        self.integral = 0.0
        self.derivativo_filtrado = 0.0
        self.v_ant = 0

    def estimacion_vel_zm(self, error, dt):
        """
        error: error que le pasamos (Er, Eh, Etheta, etc.)
        """
        
        if self.e_ant is None:
            self.e_ant = error
            derivativo_crudo = 0.0
        else:
            derivativo_crudo = (error - self.e_ant)/dt
            
        # Filtrado derivativo
        self.derivativo_filtrado = (self.alpha_d * derivativo_crudo) + \
                                    ((1 - self.alpha_d) * self.derivativo_filtrado)
        derivativo = self.derivativo_filtrado
                
        # Integral
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)   # anti-windup básico
            
        if abs(error) <= self.zm:
            v_sat = 0
            self.integral = 0.0
        else:
            v = (self.k[0] * error) + \
                (self.k[1] * derivativo) + \
                (self.k[2] * self.integral)
                
            v_sat = int(np.clip(v, -50, 50))
            
        v_sat = int(np.clip(v_sat, self.v_ant - self.max_delta, self.v_ant + self.max_delta))
        
        self.v_ant = v_sat
            
        self.e_ant = error
        
        return v_sat
