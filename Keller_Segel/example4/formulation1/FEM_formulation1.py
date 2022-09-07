#!/usr/bin/env python3
#

from Keller_Segel_equation_2d import KSData12
from fealpy.tools.show import show_error_table
from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table
from fealpy.boundarycondition import DirichletBC
from fealpy.functionspace.LagrangeFiniteElementSpace_test import LagrangeFiniteElementSpace
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.decorator import cartesian
from fealpy.mesh.HalfEdgeMesh2d_test import HalfEdgeMesh2d
from fealpy.timeintegratoralg import UniformTimeLine
import argparse

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

pde = KSData12()
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)



tmesh = UniformTimeLine(0, tend, nt)

j = 1

dt = tmesh.current_time_step_length()

steps = np.arange(0,nt+1,1)
minU = []
minV = []
space = LagrangeFiniteElementSpace(smesh, p=degree)
uh0 = space.interpolation(pde.init_valueU)
vh0 = space.function()
M = space.mass_matrix()
L = space.stiff_matrix()
bv = M@uh0
Av = L + M
vh0[:] = spsolve(Av, bv)
vh0.add_plot(plt,cmap='rainbow')
plt.savefig('v/uniformmesh/vh-' + str(j) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
plt.close()
uh0.add_plot(plt,cmap='rainbow')
plt.savefig('u/uniformmesh/uh-' + str(j) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
plt.close()


#下一个时间层的解
uh1 = space.function()
vh1 = space.function()
for k in range(nt):
    t1 = tmesh.next_time_level()
    
    bv = M@uh0
    b = np.block([bv, np.zeros(bv.size, dtype=np.float64)])
    
    #求解u
    #组装矩阵 
    K,K1 = space.ks_nonlinear_matrix(vh0)
    K += K1

    A = bmat([[M + dt*(L - K), None], [M, -(L + M)]], format='csr')
    

    #求解方程组
    x = spsolve(A, b)
    uh1[:] = x[:bv.size] 
    vh1[:] = x[bv.size:]
       
    #为下一次计算作准备
    uh0[:] = uh1
    vh0[:] = vh1
    print("t1",t1)
    if (t1 > 0.32) & (t1 < 0.6):
        uh0.add_plot(plt,cmap='rainbow')
        plt.savefig('u/uniformmesh/uh-' + str(j) +'-t1-' + str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        vh0.add_plot(plt,cmap='rainbow')
        plt.savefig('v/uniformmesh/vh-' + str(j) + '-t1-'+ str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
    if  (abs(t1 - 0.00335)< 1e-4) | (abs(t1 - 0.00667)< 1e-4) | (abs(t1 - 1.000003)< 1e-4) | (abs(t1 - 1.000003)< 1e-4):
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
