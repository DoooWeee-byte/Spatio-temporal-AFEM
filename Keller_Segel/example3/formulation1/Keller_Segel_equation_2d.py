import numpy as np

from fealpy.decorator import cartesian


class KSData0:
    """
    [-0.5, 0.5]^2
    u(x, y, t) = C_u * exp(-C_u(x^2 + y^2))*cos(2*pi*t)
    v(x, y, t) = C_v*exp(-C_v*(x^2 + (y-0.5)^2))*cos(2*pi*t)
    取参数C_u = C_v = 70
    """

    def __init__(self):
        self.box = [-0.5, 0.5, -0.5, 0.5]
        self.Cu = 70
        self.Cv = 70

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    def init_valueV(self, p):
        return self.solutionV(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        Cu = self.Cu
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = Cu*exp(-Cu*(x**2 + y**2))*cos(2*pi*t)
        return val

    @cartesian
    def solutionV(self, p, t):
        Cv = self.Cv
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = Cv*exp(-Cv*(x**2 + (y - 0.5)**2))*cos(2*pi*t)
        return val

    @cartesian
    def U_derX(self, p, t):
        Cu = self.Cu
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -2*Cu*x*self.solutionU(p, t)
        return val

    @cartesian
    def U_derXX(self, p, t):
        Cu = self.Cu
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -2*Cu*self.solutionU(p, t) - 2*Cu*x*self.U_derX(p, t)
        return val

    @cartesian
    def U_derY(self, p, t):
        Cu = self.Cu
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -2*Cu*y*self.solutionU(p, t)
        return val

    @cartesian
    def U_derYY(self, p, t):
        Cu = self.Cu
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -2*Cu*self.solutionU(p, t) - 2*Cu*y*self.U_derY(p, t)
        return val

    @cartesian
    def gradientU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        Cu = self.Cu
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -2*Cu*x*self.solutionU(p, t)
        val[..., 1] = -2*Cu*y*self.solutionU(p, t)
        return val

    @cartesian
    def gradientV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        Cv = self.Cv
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -2*Cv*x*self.solutionV(p, t)
        val[..., 1] = -2*Cv*(y - 0.5)*self.solutionV(p, t)
        return val

    @cartesian
    def V_derX(self, p, t):
        Cv = self.Cv
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -2*Cv*x*self.solutionV(p, t)
        return val

    @cartesian
    def V_derXX(self, p, t):
        Cv = self.Cv
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -2*Cv*self.solutionV(p, t) - 2*Cv*x*self.V_derX(p, t)
        return val

    @cartesian
    def V_derY(self, p, t):
        Cv = self.Cv
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -2*Cv*(y - 0.5)*self.solutionV(p, t)
        return val

    @cartesian
    def V_derYY(self, p, t):
        Cv = self.Cv
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = -2*Cv*self.solutionV(p, t) - 2*Cv*(y - 0.5)*self.V_derY(p, t)
        return val

    @cartesian
    def U_derT(self, p, t):
        Cu = self.Cu
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        val = Cu*exp(-Cu*(x**2 + y**2))*(-2*pi)*sin(2*pi*t)
        return val

    @cartesian
    def V_derT(self, p, t):
        Cv = self.Cv
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        sin = np.sin
        val = Cv*exp(-Cv*(x**2 + (y - 0.5)**2))*(-2*pi)*sin(2*pi*t)
        return val

    @cartesian
    def sourceU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        Cu = self.Cu
        Cv = self.Cv
        val = self.U_derT(p, t) - self.U_derXX(p, t) - self.U_derYY(p, t) + self.U_derX(p, t)*self.V_derX(p, t) + self.solutionU(
            p, t)*self.V_derXX(p, t) + self.U_derY(p, t)*self.V_derY(p, t) + self.solutionU(p, t)*self.V_derYY(p, t)
        return val
    '''
    @cartesian
    def gradientU(p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        Cu = self.Cu
        Cv = self.Cv
        val =
    '''
    @cartesian
    def sourceV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        Cu = self.Cu
        Cv = self.Cv
        val = self.V_derT(p, t) - self.V_derXX(p, t) - \
            self.V_derYY(p, t) - self.solutionU(p, t) + self.solutionV(p, t)
        # val = (Cv*exp(-Cv*(x**2 + (y - 0.5)**2))*(-2*pi)*sin(2*pi*t)
        #        ) - (-2*Cv*Cv*exp(-Cv*(x**2 + (y - 0.5)**2))*cos(2*pi*t) -
        #             2*Cv*x*-2*Cv*x*Cv *
        #             exp(-Cv*(x**2 + (y - 0.5)**2))*cos(2*pi*t)
        #             ) - \
        #     (-2*Cv*Cv*exp(-Cv*(x**2 + (y - 0.5)**2))*cos(2*pi*t) - 2*Cv *
        #      (y - 0.5)*-2*Cv*(y - 0.5)*Cv *
        #      exp(-Cv*(x**2 + (y - 0.5)**2))*cos(2*pi*t)
        #      ) - (Cu*exp(-Cu*(x**2 + y**2))*cos(2*pi*t)
        #           ) + (Cv*exp(-Cv*(x**2 + (y - 0.5)**2))*cos(2*pi*t)
        #                )
        return val

    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        Cu = self.Cu
        Cv = self.Cv
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = self.U_derT(p, t) - self.U_derXX(p, t) - self.U_derYY(p, t) + self.U_derX(p, t)*self.V_derX(p, t) + self.solutionU(
            p, t)*self.V_derXX(p, t) + self.U_derY(p, t)*self.V_derY(p, t) + self.solutionU(p, t)*self.V_derYY(p, t)
        val[..., 1] = self.V_derT(p, t) - self.V_derXX(p, t) - \
            self.V_derYY(p, t) - self.solutionU(p, t) + self.solutionV(p, t)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0


class KSData1:
    """
    [-0.5, 0.5]^2
    u,v unknow
    source = 0
    取参数C_u = C_v = 70
    """

    def __init__(self):
        self.box = [-0.5, 0.5, -0.5, 0.5]
        self.Cu = 70
        self.Cv = 70

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    def init_valueV(self, p):
        return self.solutionV(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        Cu = self.Cu
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = Cu*exp(-Cu*(x**2 + y**2))
        return val

    @cartesian
    def solutionV(self, p, t):
        Cv = self.Cv
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        cos = np.cos
        val = Cv*exp(-Cv*(x**2 + (y - 0.5)**2))
        return val

    @cartesian
    def sourceU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        Cu = self.Cu
        Cv = self.Cv
        val = np.zeros(p.shape[0])
        return val

    @cartesian
    def sourceV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        Cu = self.Cu
        Cv = self.Cv

        val = np.zeros()

        return val

    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        exp = np.exp
        Cu = self.Cu
        Cv = self.Cv
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0


class KSData2:
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
        val = 840*exp(-84*(x**2+y**2))
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        val = 420*exp(-42*(x**2+y**2))
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


class KSData3:
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
        val = 1200*exp(-120*(x**2+y**2))
        return val

    @cartesian
    def solutionV(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        val = 600*exp(-60*(x**2+y**2))
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


class KSData4:
    """
    [0 1]^2
    u,v unknow
    source = 0
    """

    def __init__(self):
        self.box = [0, 1, 0, 1]

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
        val = 2800*exp(-1150*((x - 0.9)**2+(y - 0.9)**2))
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


class KSData5:
    """
    [-0.5 0.5]^2
    u,v unknow
    source = 0
    """

    def __init__(self):
        self.box = [-0.5, 0.5, -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        theta = 0.01
        val = 3/theta*exp(-((x - 0.1)**2 + (y - 0.1)**2)/(2*theta))
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


class KSData6:
    """
    [-0.5 0.5]^2
    u,v unknow
    source = 0
    """

    def __init__(self):
        self.box = [-0.5, 0.5, -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        theta = 0.01
        val = 2/theta*exp(-((x - 0.1)**2 + (y - 0.1)**2)/(2*theta)) + \
            1/theta*exp(-((x + 0.2)**2 + (y + 0.2)**2)/(2*theta))

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
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0

class KSData7:
    """
    (-1,1) \times (-1/2, 1/2)
    u,v unknow
    source = 0
    """

    def __init__(self):
        self.box = [-1, 1 , -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        theta = 0.01
        val = 2/theta*exp(-((x - 0.1)**2 + (y - 0.1)**2)/(2*theta)) + \
            1/theta*exp(-((x + 0.2)**2 + (y + 0.2)**2)/(2*theta))

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
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0
        
class KSData8:
    """
    (-1/2, 1/2)^2
    u,v unknow
    source = 0
    """

    def __init__(self):
        self.box = [-0.5, 0.5 , -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        theta = 0.01
        val = 100*exp(-(x**2 + y**2)/(0.04)) + \
            60*exp(-((x - 0.2)**2 + y**2)/0.05) + 30*exp(-(x**2 + (y - 0.2)**2)/0.05)

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
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0
        
class KSData9:
    """
    (-1/2, 1/2)^2
    u,v unknow
    source = 0
    """

    def __init__(self):
        self.box = [-0.5, 0.5 , -0.5, 0.5]

    def domain(self):
        return self.box

    def init_valueU(self, p):
        return self.solutionU(p, 0.0)

    @cartesian
    def solutionU(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        theta = 0.01
        val = 40*exp(-10*(x**2 + y**2)) + 10

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
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def neuman(self, p):
        return 0.0

class KSData10:
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
        val = 1000*exp(-100*((x - 0.15)**2+(y - 0.15)**2))
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

class KSData11:
    """
    [0, 1]^2
    u,v unknow
    source = 0
    """

    def __init__(self):
        self.box = [0, 1, 0, 1]

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
        theta = 0.0003524145168463397
        val = 3.1/(pi*theta)*exp(-((x - 0.15)**2+(y - 0.15)**2)/(2*theta))
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

