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
        val = 100*exp(-(x**2 + y**2)/0.04) + 60*exp(-((x - 0.2)**2 + y**2)/0.05) + 30*exp(-(x**2 + (y - 0.2)**2)/0.05)
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

