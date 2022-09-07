#!/usr/bin/env python3
#

from Keller_Segel_equation_2d import KSData
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

parser.add_argument('--degree',
                    default=2, type=int,
                    help='拉格朗日有限元空间的维数，默认使用2次元')
args = parser.parse_args()


degree = args.degree
dim = args.dim
ns = args.ns
nt = args.nt
tend = args.T

maxit = 4

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]

errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)
pde = KSData()
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
# smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)
space = LagrangeFiniteElementSpace(smesh, p=degree)
tmesh = UniformTimeLine(0, tend, nt)
dt = tmesh.current_time_step_length()

steps = np.arange(0, nt+1, 1)

for k in range(0,maxit):
    uh0 = space.interpolation(pde.init_valueU)
    vh0 = space.interpolation(pde.init_valueV)
    M = space.mass_matrix()
    L = space.stiff_matrix()

    NDof[k] = space.number_of_global_dofs()
    # 下一个时间层的解
    uh1 = space.function()
    vh1 = space.function()
    for i in range(nt):
        t1 = tmesh.next_time_level()
        print("t1",t1)

        @cartesian
        def sourceU(p):
            return pde.sourceU(p, t1)

        @cartesian
        def sourceV(p):
            return pde.sourceV(p, t1)
        FU = space.source_vector(sourceU)
        FV = space.source_vector(sourceV)

        # 不动点迭代
        uh_new = space.function()
        vh_new = space.function()
        uh_old = space.function()
        vh_old = space.function()
        uh_old[:] = uh0
        vh_old[:] = vh0
        while True:
            AV = M + dt*L + dt*M
            bV = dt*M@uh_old[:] + M@vh0[:] + dt*FV
            '''
            @cartesian
            def neumann(p, n):
                return pde.neumann(p, n, t1)
            bc = NeumannBC(space, neumann)
            space.set_neumann_bc(F=bV, gN=neumann)
            AV, bV =bc.apply(bV, A = AV)
            '''
            vh_new[:] = spsolve(AV,bV)
            K,K1 = space.ks_nonlinear_matrix(vh_new)
            K += K1
            AU = M + dt*L - dt*K 
            bU = M@uh0[:] + dt*FU
            '''
            space.set_neumann_bc(F=bU, gN=neumann)
            AU, bU =bc.apply(bU, A = AU)
            '''
            uh_new[:] = spsolve(AU, bU)
            print("errorU",np.linalg.norm(uh_new - uh_old))
            print("errorV",np.linalg.norm(vh_new - vh_old))
            if(np.linalg.norm(uh_new - uh_old) < 1e-8) & (np.linalg.norm(vh_new - vh_old) < 1e-8):
                break
            uh_old[:] = uh_new[:]
            vh_old[:] = vh_new[:]


        uh1[:] = uh_new[:]
        vh1[:] = vh_new[:]


        # t1 时间层的误差
        @cartesian
        def solutionU(p):
            return pde.solutionU(p, t1)

        @cartesian
        def solutionV(p):
            return pde.solutionV(p, t1)

        @cartesian
        def gradientU(p):
            return pde.gradientU(p, t1)

        @cartesian
        def gradientV(p):
            return pde.gradientV(p, t1)

        uh0[:] = uh1
        vh0[:] = vh1
        print("L2error:",space.integralalg.error(solutionU, uh1))
        if i == nt-1:
            errorMatrix[0, k] = space.integralalg.error(solutionU, uh1)
            errorMatrix[1, k] = space.integralalg.error(
                gradientU, uh1.grad_value)

        if (i == nt-1) & (k == maxit-1):
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1, projection='3d')
            uh1.add_plot(axes, cmap='rainbow')
        tmesh.advance()
    if k < maxit-1:
        print("ns", ns)
        ns += 2
        smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
        #smesh.uniform_refine()
        space = LagrangeFiniteElementSpace(smesh, p=degree)
        tmesh = UniformTimeLine(0, tend, nt)
        print("----------------")
        print(space.number_of_global_dofs())

show_error_table(NDof, errorType, errorMatrix, out='first.txt')
showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=35)
plt.show()
