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

class PolyData:
    """

    u_t - \\Delta u + \\epsilon^{-2} f(u)  = F

    \\epsilon = 0.02
    f(u) = u^3 - u
    u(x, y, t) = exp(t)*x^2*(x - 1)^2*y^2*(y - 1)^2
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
        self.e = 0.02

    def domain(self):
        return [0, 1, 0, 1]

    def init_value(self, p):
        return self.solution(p, 0.0)

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        u = np.exp(t)*x**2*(x - 1)**2*y**2*(y - 1)**2
        return u

    def u(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        result = self.solution(p, t)
        return result

    def u_t(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        e = self.e
        u = np.exp(t)*x**2*(x - 1)**2*y**2*(y - 1)**2
        return u

    def u_x(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.exp(t)*(4*x**3 + 2*x - 6*x**2)*(y**4 + y**2 -2*y**3)
        return u

    def u_y(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.exp(t)*(4*y**3 + 2*y - 6*y**2)*(x**4 + x**2 -2*x**3)
        return u

    def u_xx(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.exp(t)*(12*x**2 + 2 - 12*x)*(y**4 + y**2 - 2*y**3)
        return u

    def u_yy(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.exp(t)*(12*y**2 + 2 - 12*y)*(x**4 + x**2 - 2*x**3)
        return u

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        e = self.e
        rhs = self.u_t(p,t) - self.u_xx(p,t) - self.u_yy(p, t) + e**(-2)*self.u(p,t)**3 - e**(-2)*self.u(p,t)
        return rhs

    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.u_x(p, t)
        val[..., 1] = self.u_y(p, t)
        return val

    def neuman(self, p, t):
        pass

    def is_neuman_boundary(self, p):
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
        self.e = 0.02

    def domain(self):
        return [-1, 1, -1, 1]

    def init_value(self, p):
        return self.solution(p, 0.0)
        
    def u(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        result = self.solution(p, t)
        return result


    def beta(self, t):
        return 0.1*(1 - np.exp(-10000*(t - 0.5)**2))

    def alpha(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.exp(-((x-1000*t+0.5)**2+(y-1000*t+0.5)**2)/0.04)
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
        u = 2000*(t - 0.5)*exp(-10000*(t-0.5)**2)*self.alpha(p, t) + 1/0.02*(x + y - 2000*t + 1)*1000*self.alpha(p,t)*self.beta(t)
        return u

    def u_x(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*(x - 1000*t +0.5)*self.alpha(p,t)*self.beta(t)
        return u

    def u_y(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*(y - 1000*t + 0.5)*self.alpha(p,t)*self.beta(t)
        return u

    def u_xx(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*self.alpha(p,t)*self.beta(t) - 1/0.02*(x - 1000*t + 0.5)*self.beta(t)*(-1/0.02*(x - 1000*t +0.5)*self.alpha(p,t))
        return u

    def u_yy(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -1/0.02*self.alpha(p,t)*self.beta(t) - 1/0.02*(y - 1000*t + 0.5)*self.beta(t)*(-1/0.02*(y - 1000*t +0.5)*self.alpha(p,t))
        return u

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        e = self.e
        rhs = self.u_t(p,t) - self.u_xx(p,t) - self.u_yy(p, t) + e**(-2)*self.u(p,t)**3 - e**(-2)*self.u(p,t)
        return rhs

    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.u_x(p, t)
        val[..., 1] = self.u_y(p, t)
        return val

    def neuman(self, p, t):
        pass

    def is_neuman_boundary(self, p):
        eps = 1e-14
        return (p[..., 0] < eps) | (p[..., 1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)
 
class ExpCosCos:
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
        self.e = 1

    def domain(self):
        return [-1, 1, -1, 1]

    def init_value(self, p):
        return self.solution(p, 0.0)
        
    def u(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        result = self.solution(p, t)
        return result

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.exp(t)*np.cos(pi*x)*np.cos(pi*y)
        return u

    def u_t(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        exp = np.exp
        u = np.exp(t)*np.cos(pi*x)*np.cos(pi*y)
        return u

    def u_x(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -pi*np.exp(t)*np.sin(pi*x)*np.cos(pi*y)
        return u

    def u_y(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -pi*np.exp(t)*np.sin(pi*y)*np.cos(pi*x)
        return u

    def u_xx(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -pi*pi*np.exp(t)*np.cos(pi*x)*np.cos(pi*y)
        return u

    def u_yy(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = -pi*pi*np.exp(t)*np.cos(pi*x)*np.cos(pi*y)
        return u

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        e = self.e
        rhs = self.u_t(p,t) - self.u_xx(p,t) - self.u_yy(p, t) + e**(-2)*self.u(p,t)**3 - e**(-2)*self.u(p,t)
        return rhs

    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = self.u_x(p, t)
        val[..., 1] = self.u_y(p, t)
        return val

    def neuman(self, p, t):
        pass

    def is_neuman_boundary(self, p):
        eps = 1e-14
        return (p[..., 0] < eps) | (p[..., 1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps) 
   
class TanhData:
    '''
    '''
    def init_mesh(self, n=1, meshtype='tri'):
        node = np.array([
            (-2, -2),
            (2, -2),
            (2, 2),
            (-2, 2)], dtype=np.float64)

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
        self.e = 1.0/16

    def domain(self):
        return [-2, 2, -2, 2]

    def init_value(self, p):
        x = p[..., 0]
        y = p[..., 1]
        r = np.sqrt(x**2 + y**2)
        u = np.zeros(x.shape)
        u[r<0.7] = np.tanh(16/np.sqrt(2)*(r[r<0.7] - 0.4))
        u[r>=0.7] = -np.tanh(16/np.sqrt(2)*(r[r>=0.7] - 1.0))
        return u

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        rhs = x*0
        return rhs
 
    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        r = np.sqrt(x**2 + y**2)
        u = np.zeros(x.shape)
        u[r<0.7] = np.tanh(16/np.sqrt(2)*(r[r<0.7] - 0.4))
        u[r>=0.7] = -np.tanh(16/np.sqrt(2)*(r[r>=0.7] - 1.0))
        return u
       
    def neuman(self, p, t):
        pass

    def is_neuman_boundary(self, p):
        eps = 1e-14
        return (p[..., 0] < eps) | (p[..., 1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)
