import numpy as np

from fealpy.decorator import cartesian


class KSData:
    """
    [-0.5, 0.5]^2
    u(x, y, t) = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
    v(x, y, t) = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
    """

    def __init__(self):
        self.box = [-0.5, 0.5, -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    def init_valueV(self, p):
        return self.solutionV(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def U_derX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -pi*exp(-2*pi**2*t)*sin(pi*x)*cos(pi*y)
        return val

    @cartesian
    def U_derXX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def U_derY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -pi*exp(-2*pi**2*t)*cos(pi*x)*sin(pi*y)
        return val

    @cartesian
    def U_derYY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def gradientU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*exp(-2*pi**2*t)*sin(pi*x)*cos(pi*y)
        val[..., 1] = -pi*exp(-2*pi**2*t)*cos(pi*x)*sin(pi*y)
        return val

    @cartesian
    def gradientV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*exp(-2*pi**2*t)*sin(pi*x)*cos(pi*y)
        val[..., 1] = -pi*exp(-2*pi**2*t)*cos(pi*x)*sin(pi*y)
        return val

    @cartesian
    def V_derX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -pi*exp(-2*pi**2*t)*sin(pi*x)*cos(pi*y)
        return val

    @cartesian
    def V_derXX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def V_derY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -pi*exp(-2*pi**2*t)*cos(pi*x)*sin(pi*y)
        return val

    @cartesian
    def V_derYY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def U_derT(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        cos = np.cos
        val = -2*pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def V_derT(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        cos = np.cos
        val = -2*pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def sourceU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        """
        val = (self.U_derX(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derX(p, t))/(1 + self.solutionV(p, t))**2*self.V_derX(p, t) + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derXX(p, t) + \
            (self.U_derY(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derY(p, t))/(1 +
                                                                                                     self.solutionV(p, t))**2*self.V_derY(p, t) + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derYY(p, t)
        """
        val = self.U_derT(p, t) - self.U_derXX(p, t) - self.U_derYY(p, t) + (self.U_derX(p, t)*(1 + self.solutionV(p, t)) \
                - self.solutionU(p, t)*self.V_derX(p, t))/(1 + self.solutionV(p, t))**2*self.V_derX(p, t) + \
            self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derXX(p, t) \
            + (self.U_derY(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derY(p, t))/(1 + self.solutionV(p, t))**2*self.V_derY(p, t) \
            + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derYY(p, t)

        return val

    @cartesian
    def sourceV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = self.V_derT(p,t)  -self.V_derXX(p, t) - self.V_derYY(p,t) -self.U_derXX(p, t) - self.U_derYY(p, t) - \
            self.solutionU(p, t) + self.solutionV(p, t)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0

class KSData1:
    """
    [-0.5, 0.5]^2
    u(x, y, t) = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
    v(x, y, t) = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
    """

    def __init__(self):
        self.box = [-0.5, 0.5, -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    def init_valueV(self, p):
        return self.solutionV(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def U_derX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -pi*exp(-2*pi**2*t)*sin(pi*x)*cos(pi*y)
        return val

    @cartesian
    def U_derXX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def U_derY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -pi*exp(-2*pi**2*t)*cos(pi*x)*sin(pi*y)
        return val

    @cartesian
    def U_derYY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def gradientU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*exp(-2*pi**2*t)*sin(pi*x)*cos(pi*y)
        val[..., 1] = -pi*exp(-2*pi**2*t)*cos(pi*x)*sin(pi*y)
        return val

    @cartesian
    def gradientV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*exp(-2*pi**2*t)*sin(pi*x)*cos(pi*y)
        val[..., 1] = -pi*exp(-2*pi**2*t)*cos(pi*x)*sin(pi*y)
        return val

    @cartesian
    def V_derX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -pi*exp(-2*pi**2*t)*sin(pi*x)*cos(pi*y)
        return val

    @cartesian
    def V_derXX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def V_derY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -pi*exp(-2*pi**2*t)*cos(pi*x)*sin(pi*y)
        return val

    @cartesian
    def V_derYY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def U_derT(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        cos = np.cos
        val = -2*pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def V_derT(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        cos = np.cos
        val = -2*pi**2*exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
        return val

    @cartesian
    def sourceU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        """
        val = (self.U_derX(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derX(p, t))/(1 + self.solutionV(p, t))**2*self.V_derX(p, t) + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derXX(p, t) + \
            (self.U_derY(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derY(p, t))/(1 +
                                                                                                     self.solutionV(p, t))**2*self.V_derY(p, t) + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derYY(p, t)
        """
        val = self.U_derX(p,t)*self.V_derX(p,t) + self.solutionU(p,t)*self.V_derXX(p,t) \
            + self.U_derY(p,t)*self.V_derY(p,t) + self.solutionU(p,t)*self.V_derYY(p,t)
        return val

    @cartesian
    def sourceV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = self.V_derT(p,t)  -self.V_derXX(p, t) - self.V_derYY(p,t) -self.U_derXX(p, t) - self.U_derYY(p, t) - \
            self.solutionU(p, t) + self.solutionV(p, t)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0
        
class KSData2:
    """
    [-0.5, 0.5]^2
    u(x, y, t) = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
    v(x, y, t) = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
    """

    def __init__(self):
        self.box = [-0.5, 0.5, -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    def init_valueV(self, p):
        return self.solutionV(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = 0.8*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = 0.8*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def U_derX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -0.8*2*pi*exp(-2*pi**2*t)*sin(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def U_derXX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -0.8*4*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def U_derY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -0.8*2*pi*exp(-2*pi**2*t)*cos(2*pi*x)*sin(2*pi*y)
        return val

    @cartesian
    def U_derYY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -0.8*4*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def gradientU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -2*0.8*pi*exp(-2*pi**2*t)*sin(2*pi*x)*cos(2*pi*y)
        val[..., 1] = -2*0.8*pi*exp(-2*pi**2*t)*cos(2*pi*x)*sin(2*pi*y)
        return val

    @cartesian
    def gradientV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -2*0.8*pi*exp(-2*pi**2*t)*sin(2*pi*x)*cos(2*pi*y)
        val[..., 1] = -2*0.8*pi*exp(-2*pi**2*t)*cos(2*pi*x)*sin(2*pi*y)
        return val

    @cartesian
    def V_derX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -2*0.8*pi*exp(-2*pi**2*t)*sin(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def V_derXX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -4*0.8*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def V_derY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -2*0.8*pi*exp(-2*pi**2*t)*cos(2*pi*x)*sin(2*pi*y)
        return val

    @cartesian
    def V_derYY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -4*0.8*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def U_derT(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        cos = np.cos
        val = -2*0.8*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def V_derT(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        cos = np.cos
        val = -2*0.8*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def sourceU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        """
        val = (self.U_derX(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derX(p, t))/(1 + self.solutionV(p, t))**2*self.V_derX(p, t) + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derXX(p, t) + \
            (self.U_derY(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derY(p, t))/(1 +
                                                                                                     self.solutionV(p, t))**2*self.V_derY(p, t) + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derYY(p, t)
        """
        val = self.U_derT(p, t) - self.U_derXX(p, t) - self.U_derYY(p, t) + (self.U_derX(p, t)*(1 + self.solutionV(p, t)) \
                - self.solutionU(p, t)*self.V_derX(p, t))/(1 + self.solutionV(p, t))**2*self.V_derX(p, t) + \
            self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derXX(p, t) \
            + (self.U_derY(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derY(p, t))/(1 + self.solutionV(p, t))**2*self.V_derY(p, t) \
            + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derYY(p, t)

        return val

    @cartesian
    def sourceV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = self.V_derT(p,t)  -self.V_derXX(p, t) - self.V_derYY(p,t) -self.U_derXX(p, t) - self.U_derYY(p, t) - \
            self.solutionU(p, t) + self.solutionV(p, t)
        return val
        
    @cartesian
    def neumann(self, p, n, t):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradientU(p, t) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return x == 0.5 | x == -0.5 | y == 0.5 | y == -0.5
        
class KSData3:
    """
    [-0.5, 0.5]^2
    u(x, y, t) = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
    v(x, y, t) = exp(-2*pi**2*t)*cos(pi*x)*cos(pi*y)
    """

    def __init__(self):
        self.box = [-0.5, 0.5, -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    def init_valueV(self, p):
        return self.solutionV(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = 0.8*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = 0.8*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def U_derX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -0.8*2*pi*exp(-2*pi**2*t)*sin(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def U_derXX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -0.8*4*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def U_derY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -0.8*2*pi*exp(-2*pi**2*t)*cos(2*pi*x)*sin(2*pi*y)
        return val

    @cartesian
    def U_derYY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -0.8*4*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def gradientU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -2*0.8*pi*exp(-2*pi**2*t)*sin(2*pi*x)*cos(2*pi*y)
        val[..., 1] = -2*0.8*pi*exp(-2*pi**2*t)*cos(2*pi*x)*sin(2*pi*y)
        return val

    @cartesian
    def gradientV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -2*0.8*pi*exp(-2*pi**2*t)*sin(2*pi*x)*cos(2*pi*y)
        val[..., 1] = -2*0.8*pi*exp(-2*pi**2*t)*cos(2*pi*x)*sin(2*pi*y)
        return val

    @cartesian
    def V_derX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -2*0.8*pi*exp(-2*pi**2*t)*sin(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def V_derXX(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -4*0.8*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def V_derY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        sin = np.sin
        val = -2*0.8*pi*exp(-2*pi**2*t)*cos(2*pi*x)*sin(2*pi*y)
        return val

    @cartesian
    def V_derYY(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -4*0.8*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def U_derT(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        cos = np.cos
        val = -2*0.8*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def V_derT(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        cos = np.cos
        val = -2*0.8*pi**2*exp(-2*pi**2*t)*cos(2*pi*x)*cos(2*pi*y)
        return val

    @cartesian
    def sourceU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        """
        val = (self.U_derX(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derX(p, t))/(1 + self.solutionV(p, t))**2*self.V_derX(p, t) + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derXX(p, t) + \
            (self.U_derY(p, t)*(1 + self.solutionV(p, t)) - self.solutionU(p, t)*self.V_derY(p, t))/(1 +
                                                                                                     self.solutionV(p, t))**2*self.V_derY(p, t) + self.solutionU(p, t)/(1 + self.solutionV(p, t))*self.V_derYY(p, t)
        """
        val = self.U_derT(p, t) - self.U_derXX(p, t) - self.U_derYY(p, t) + (self.U_derX(p,t)*self.V_derX(p,t) + self.solutionU(p,t)*self.V_derXX(p,t) + \
            self.U_derY(p ,t)*self.V_derY(p ,t) + self.solutionU(p,t)*self.V_derYY(p,t))

        return val

    @cartesian
    def sourceV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = self.V_derT(p,t)  -self.V_derXX(p, t) - self.V_derYY(p,t) + self.solutionU(p, t) - self.solutionV(p, t)
        return val
        
    @cartesian
    def neumann(self, p, n, t):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradientU(p, t) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return x == 0.5 | x == -0.5 | y == 0.5 | y == -0.5
