import numpy as np

from fealpy.decorator import cartesian

class KSData:
    """
    [-0.5, 0.5]^2
    u,v unknow
    source = 0
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
        val = 40*exp(-10*(x**2 + y**2)) + 10 
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        val = 0*x
        return val

    @cartesian
    def sourceU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape[0])
        return val

    @cartesian
    def sourceV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros()

        return val

    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0
class KSData1:
    """
    [0, 2*pi]^2
    u(x, y, t) = exp(-t)(cos(x) + cos(y))
    v(x, y, t) = exp(-t)(cos(x) + cos(y))
    """

    def __init__(self):
        pi = np.pi
        self.box = [0, 2*pi, 0, 2*pi]

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
        val = exp(-t)*(cos(x) + cos(y))
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = exp(-t)*(cos(x) + cos(y))
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
        val[..., 0] = exp(-t)*(-sin(x))
        val[..., 1] = exp(-t)*(-sin(y))
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
        val[..., 0] = exp(-t)*(-sin(x))
        val[..., 1] = exp(-t)*(-sin(y))
        return val


    @cartesian
    def sourceU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        #val = -2*exp(-2*t)*(cos(x)**2 + cos(x)*cos(y) + cos(y)**2 - 1)
        val = exp(-t)**2*(sin(x)**2 + sin(y)**2 - cos(x)**2 - cos(y)**2 - 2*cos(x)*cos(y))
        return val

    @cartesian
    def sourceV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        val = exp(-t)*(cos(x) + cos(y))
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
        pi = np.pi
        return x == 0 | x == 2*pi | y == 0 | y == 2*pi

