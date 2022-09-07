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
        val = 1000*exp(-100*(x**2+y**2))
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        val = 500*exp(-50*(x**2+y**2))
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
        val = np.sum(grad*n, axis=-1)*0 # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return x == 0.5 | x == -0.5 | y == 0.5 | y == -0.5


