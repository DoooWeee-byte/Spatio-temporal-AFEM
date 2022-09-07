#!/usr/bin/env python3
#

from Keller_Segel_equation_2d import KSData2
from fealpy.tools.show import show_error_table
from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table
from fealpy.boundarycondition import DirichletBC
from fealpy.functionspace.LagrangeFiniteElementSpace import LagrangeFiniteElementSpace
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.decorator import cartesian
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
from fealpy.timeintegratoralg import UniformTimeLine
import argparse
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# 装饰子：指明被装饰函数输入的是笛卡尔坐标点

# 网格工厂：生成常用的简单区域上的网格

# 均匀剖分的时间离散

# 热传导 pde 模型

# Lagrange 有限元空间

# Dirichlet 边界条件
# solver

# 画图


# 参数解析
parser = argparse.ArgumentParser(description="""
        单纯形网格（三角形、四面体）网格上任意次有限元方法求解热传导方程
        """)

parser.add_argument('--dim',
                    default=2, type=int,
                    help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--ns',
                    default=10, type=int,
                    help='空间各个方向剖分段数， 默认剖分 10 段.')

parser.add_argument('--nt',
                    default=100, type=int,
                    help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--T',
                    default=0.5, type=float,
                    help='终止时间，默认为 0.5 .')

args = parser.parse_args()

degree = 1
dim = args.dim
ns = args.ns
nt = args.nt
tend = args.T

pde = KSData2()
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)
smesh.add_plot(plt)
plt.savefig('mesh/test' + str(0) + '.png')
plt.close()


#定义函数组装非线性项形成的矩阵
def ks_nonlinear_matrix(uh,q=None):
    '''
    (u\\nabla v, \\phi)
    v是未知函数，\\phi是test function，u是已知函数
    '''
    space = uh.space
    mesh = space.mesh
    GD = mesh.geo_dimension()
    qf = space.integrator if q is None else mesh.integrator(q, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cellmeasure = space.cellmeasure
    gdof = space.number_of_global_dofs()
    c2d = space.cell_to_dof()
    ugrad = uh.grad_value(bcs)
    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    shape = c2d.shape + c2d.shape[1:]
    I = np.broadcast_to(c2d[:, :, None], shape=shape)
    J = np.broadcast_to(c2d[:, None, :], shape=shape)
    B = []
    for i in range(2):
        val = np.einsum('q, qc, qci, qcj, c->cij', ws,
                        ugrad[..., i], gphi[..., i], phi, cellmeasure)
        M = csr_matrix(
            (val.flat, (I.flat, J.flat)),
            shape=(gdof, gdof)
        )
        B += [M]
    return B

tmesh = UniformTimeLine(0, tend, nt)

j = 1

dt = tmesh.current_time_step_length()

steps = np.arange(0,nt+1,1)
minU = []
minV = []
space = LagrangeFiniteElementSpace(smesh, p=degree)
uh0 = space.interpolation(pde.init_valueU)
vh0 = space.interpolation(pde.init_valueV)
uh0.add_plot(plt,cmap='rainbow')
plt.savefig('u/uniformmesh/uh-' + str(j) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
plt.close()
vh0.add_plot(plt,cmap='rainbow')
plt.savefig('v/uniformmesh/vh-' + str(j) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
plt.close()

M = space.mass_matrix()
L = space.stiff_matrix()
M = spdiags(np.sum(M, axis=1).flat,0,M.shape[0],M.shape[1])
#外插法准备向量
vh_lin = space.function()
vh_lin[:] = vh0 
for k in range(nt):
    t1 = tmesh.next_time_level()
    
    #下一个时间层的解
    uh1 = space.function()
    vh1 = space.function()

    #组装矩阵 
    K,K1 = ks_nonlinear_matrix(vh_lin)
    K += K1

    D = -K.copy()
    D[(D -(D.transpose()))<0] = D[(D -(D.transpose()))<0]*0  + D.transpose()[(D -(D.transpose()))<0]
    
    D -= spdiags(D.diagonal(),0,K.shape[0],K.shape[1])
    D[D<0] = 0
    D -= spdiags(np.sum(D, axis=1).flat, 0, D.shape[0],D.shape[1])

    

    A = M + dt/2*(L - K -D)
    '''
    A = M + dt/2*(L - K)
    '''
    #组装右端项

    b1 = (M - dt/2*(L - K - D))@uh0[:]
    '''
    b1 = (M - dt/2*(L - K))@uh0[:]
    '''
    #求解方程组
    uh1[:] = spsolve(A, b1)

    A1 = M + dt/2*L + dt/2*M
    F = dt/2*M@(uh1[:] + uh0[:])
    b2 = (M - dt/2*L - dt/2*M)@vh0[:] + F

    vh1[:] = spsolve(A1, b2)
    vh_lin[:] = (3*vh1 - vh0)/2 
       
    #为下一次计算作准备
    uh0[:] = uh1
    vh0[:] = vh1

    if (abs(t1 - tend)< 1e-8):
        uh0.add_plot(plt,cmap='rainbow')
        plt.savefig('u/uniformmesh/uh-' + str(j) +'-t1-' + str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        vh0.add_plot(plt,cmap='rainbow')
        plt.savefig('v/uniformmesh/vh-' + str(j) + '-t1-'+ str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        uh0.add_plot(axes, cmap='rainbow')
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        vh0.add_plot(axes, cmap='rainbow')
        
        minU.append(min(uh0[:]))
        minV.append(min(vh0[:]))
        j += 1
    tmesh.advance()

plt.show()
