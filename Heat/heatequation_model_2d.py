import numpy as np
from fealpy.mesh.Tritree import Tritree
from fealpy.decorator import cartesian, barycentric
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.mesh.Tritree import Tritree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityNode
from fealpy.mesh.PolygonMesh import PolygonMesh
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d

class ExpCosData:
    """

    u_t - c*\Delta u = f

    c = 1
    u(x, y, t) = exp(-a((x-0.5)^2 + (y-0.5)^2))*cos(2*pi*t)

    domain = [0, 1]^2


    """
    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            mesh = Tritree(node, cell)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))



    def __init__(self):
        self.diffusionCoefficient = 1
        self.a = 100

    def domain(self):
        return [0, 1, 0, 1]

    def init_value(self, p):
        return self.solution(p, 0.0)

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        a = self.a
        u = np.exp(-a*((x - 0.5)**2 + (y - 0.5)**2))*np.cos(2*pi*t)
        return u

    def u_t(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        a = self.a
        u = np.exp(-a*((x - 0.5)**2 + (y - 0.5)**2))*np.sin(2*pi*t)*(-2*pi)
        return u

    def u_x(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        a = self.a
        u = -2*a*(x - 0.5)*np.exp(-a*((x - 0.5) **
                                      2 + (y - 0.5)**2))*np.cos(2*pi*t)
        return u

    def u_y(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        a = self.a
        u = -2*a*(y - 0.5)*np.exp(-a*((x - 0.5) **
                                      2 + (y - 0.5)**2))*np.cos(2*pi*t)
        return u

    def u_xx(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        a = self.a
        u = (-2*a*np.exp(-a*((x - 0.5)**2 + (y - 0.5)**2)) + 4*a*a *
             (x - 0.5)**2*np.exp(-a*((x - 0.5)**2 + (y - 0.5)**2)))*np.cos(2*pi*t)
        return u

    def u_yy(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        a = self.a
        u = (-2*a*np.exp(-a*((x - 0.5)**2 + (y - 0.5)**2)) + 4*a*a *
             (y - 0.5)**2*np.exp(-a*((x - 0.5)**2 + (y - 0.5)**2)))*np.cos(2*pi*t)
        return u

    def diffusion_coefficient(self, p):
        return self.diffusionCoefficient

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        k = self.diffusionCoefficient
        rhs = self.u_t(p, t) - k*(self.u_xx(p, t) + self.u_yy(p, t))
        return rhs

    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.u_x(p, t)
        val[..., 1] = self.u_y(p, t)
        return val

    def dirichlet(self, p, t):
        return self.solution(p, t)

    def is_dirichlet_boundary(self, p):
        eps = 1e-14
        return (p[..., 0] < eps) | (p[..., 1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)

class ExpExpData:
    '''
    '''
    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)], dtype=np.float64)

        if meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            mesh = Tritree(node, cell)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))



    def __init__(self):
        self.diffusionCoefficient = 1

    def domain(self):
        return [-1, 1, -1, 1]

    def init_value(self, p):
        return self.solution(p, 0.0)

    def beta(self, t):
        return 0.1*(1 - np.exp(-10000*(t - 0.5)**2))

    def alpha(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.exp(-((x-t+0.5)**2+(y-t+0.5)**2)/0.04)
        return u

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = self.beta(t)*self.alpha(p, t)
        return u

    def u_t(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        u = 2000*(t - 0.5)*exp(-10000*(t-0.5)**2)*self.alpha(p, t) + 1/0.02*(x + y - 2*t + 1)*self.alpha(p,t)*self.beta(t)
        return u

    def u_x(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*(x - t +0.5)*self.alpha(p,t)*self.beta(t)
        return u

    def u_y(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*(y - t + 0.5)*self.alpha(p,t)*self.beta(t)
        return u

    def u_xx(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*self.alpha(p,t)*self.beta(t) - 1/0.02*(x - t + 0.5)*self.beta(t)*(-1/0.02*(x - t +0.5)*self.alpha(p,t))
        return u

    def u_yy(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*self.alpha(p,t)*self.beta(t) - 1/0.02*(y - t + 0.5)*self.beta(t)*(-1/0.02*(y - t +0.5)*self.alpha(p,t))
        return u

    def diffusion_coefficient(self, p):
        return self.diffusionCoefficient

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        k = self.diffusionCoefficient
        rhs = self.u_t(p, t) - k*(self.u_xx(p, t) + self.u_yy(p, t))
        return rhs

    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.u_x(p, t)
        val[..., 1] = self.u_y(p, t)
        return val

    def dirichlet(self, p, t):
        return self.solution(p, t)

    def is_dirichlet_boundary(self, p):
        eps = 1e-14
        return (p[..., 0] - 1  < eps) | (p[..., 1] - 1< eps) | (p[..., 0]  + 1 < eps) | (p[..., 1] + 1 < eps)
        
class ExpData:
    '''
    u(x,y,0) = 840exp(-84(x**2 + y**2))
    '''
    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-0.5, -0.5),
            (0.5, -0.5),
            (0.5, 0.5),
            (-0.5, 0.5)], dtype=np.float64)

        if meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'quadtree':
            cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
            mesh = Quadtree(node, cell)
            mesh.uniform_refine(n)
            return mesh
        elif meshtype == 'tritree':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            mesh = Tritree(node, cell)
            return mesh
        else:
            raise ValueError("I don't know the meshtype %s".format(meshtype))



    def __init__(self):
        self.diffusionCoefficient = 1

    def domain(self):
        return [-0.5, 0.5, -0.5, 0.5]

    def init_value(self, p):
        return self.solution(p, 0.0)

    def beta(self, t):
        return 0.1*(1 - np.exp(-10000*(t - 0.5)**2))

    def alpha(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.exp(-((x-t+0.5)**2+(y-t+0.5)**2)/0.04)
        return u

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = self.beta(t)*self.alpha(p, t)
        return u

    def u_t(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        u = 2000*(t - 0.5)*exp(-10000*(t-0.5)**2)*self.alpha(p, t) + 1/0.02*(x + y - 2*t + 1)*self.alpha(p,t)*self.beta(t)
        return u

    def u_x(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*(x - t +0.5)*self.alpha(p,t)*self.beta(t)
        return u

    def u_y(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*(y - t + 0.5)*self.alpha(p,t)*self.beta(t)
        return u

    def u_xx(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*self.alpha(p,t)*self.beta(t) - 1/0.02*(x - t + 0.5)*self.beta(t)*(-1/0.02*(x - t +0.5)*self.alpha(p,t))
        return u

    def u_yy(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*self.alpha(p,t)*self.beta(t) - 1/0.02*(y - t + 0.5)*self.beta(t)*(-1/0.02*(y - t +0.5)*self.alpha(p,t))
        return u

    def diffusion_coefficient(self, p):
        return self.diffusionCoefficient

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        k = self.diffusionCoefficient
        rhs = self.u_t(p, t) - k*(self.u_xx(p, t) + self.u_yy(p, t))
        return rhs

    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.u_x(p, t)
        val[..., 1] = self.u_y(p, t)
        return val

    def dirichlet(self, p, t):
        return self.solution(p, t)

    def is_dirichlet_boundary(self, p):
        eps = 1e-14
        return (p[..., 0] - 1  < eps) | (p[..., 1] - 1< eps) | (p[..., 0]  + 1 < eps) | (p[..., 1] + 1 < eps)

