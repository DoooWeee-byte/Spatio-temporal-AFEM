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

parser.add_argument('--tol',
                    default=0.05, type=float,
                    help='容许误差，默认为 0.05 .')

parser.add_argument('--tol1',
                    default=0.05, type=float,
                    help='容许误差，默认为 0.05 .')
args = parser.parse_args()

degree = 1
dim = args.dim
ns = args.ns
nt = args.nt
tend = args.T
tol = args.tol
tol1 = args.tol1
theta = 0.9    
ctheta = 0.3

pde = KSData()
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)
space = LagrangeFiniteElementSpace(smesh, p=degree)
smesh.add_plot(plt)
plt.savefig('mesh/test' + str(0) + '.png')
plt.close()


tmesh = UniformTimeLine(0, tend, nt)

vh0 = space.function()
vh0.add_plot(plt,cmap='rainbow')
plt.savefig('v/AFEM/vh-' + str(0) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
j = 1
while True:
    space = LagrangeFiniteElementSpace(smesh, p=degree)
    uh0 = space.interpolation(pde.init_valueU)
    etaU = space.recovery_estimate(uh0, method='area_harmonic')
    errU = np.sqrt(np.sum(etaU**2))
    #求数值解的梯度
    bcs, ws = space.integrator.get_quadrature_points_and_weights()
    cellmeasure = space.cellmeasure 
    uh_grad_value = uh0.grad_value(bcs)**2  #(NQ,NC,2)
    uh_grad_value = np.sum(uh_grad_value,axis=2)
    uh_grad_norm = np.einsum('q,qc,c->', ws,uh_grad_value,cellmeasure)
    uh_grad_norm = np.sqrt(uh_grad_norm)
    
    print('errU',errU)
    print('uh_grad_norm',uh_grad_norm)
    print('errU/uh_grad_norm ',errU/uh_grad_norm )
    print('dof',space.number_of_global_dofs())
    isMarkedCellU = smesh.refine_marker(etaU, theta, method='MAX')
    isMarkedCell = isMarkedCellU
    smesh.refine_triangle_rg(isMarkedCell)
    smesh.add_plot(plt)
    plt.savefig('mesh/test-' + str(j) + '-dof-'+ str(space.number_of_global_dofs()) +  '.png')
    plt.close()
    j += 1
    if (errU/uh_grad_norm < tol):
        break;

dt = tmesh.current_time_step_length()

steps = np.arange(0,nt+1,1)
minU = []
minV = []
space = LagrangeFiniteElementSpace(smesh, p=degree)
uh0 = space.interpolation(pde.init_valueU)
vh0 = space.function()
uh0.add_plot(plt,cmap='rainbow')
plt.savefig('u/AFEM/uh-' + str(j) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
plt.close()
M = space.mass_matrix()
L = space.stiff_matrix()
bv = M@uh0
Av = L + M
vh0[:] = spsolve(Av, bv)
vh0.add_plot(plt,cmap='rainbow')
plt.savefig('v/AFEM/vh-' + str(j) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
plt.close()
minU.append(min(uh0[:]))
minV.append(min(vh0[:]))
j += 1

for k in range(nt):
    t1 = tmesh.next_time_level()
    refine_count=0
    while True:
        gdof = space.number_of_global_dofs()
        M = space.mass_matrix()
        L = space.stiff_matrix()
        M = spdiags(np.sum(M, axis=1).flat,0,M.shape[0],M.shape[1])
        #下一个时间层的解
        uh1 = space.function()
        vh1 = space.function()
        
        
        bv = M@uh0
        b = np.block([bv, np.zeros(bv.size, dtype=np.float64)])

        #组装矩阵 
        K,K1 = space.ks_nonlinear_matrix(vh0)
        K += K1

        D = -K.copy()
        D[(D -(D.transpose()))<0] = D[(D -(D.transpose()))<0]*0  + D.transpose()[(D -(D.transpose()))<0]
        
        D -= spdiags(D.diagonal(),0,K.shape[0],K.shape[1])
        D[D<0] = 0
        D -= spdiags(np.sum(D, axis=1).flat, 0, D.shape[0],D.shape[1])

        A = bmat([[M + dt*(L - K - D), None], [M, -(L + M)]], format='csr')
    

        #求解方程组
        x = spsolve(A, b)
        uh1[:] = x[:bv.size] 
        vh1[:] = x[bv.size:]

        #计算误差估计子
        etaU = space.recovery_estimate(uh1, method='area_harmonic')
        etaV = space.recovery_estimate(vh1, method='area_harmonic')
        errU = np.sqrt(np.sum(etaU**2))
        errV = np.sqrt(np.sum(etaV**2))
        
        #计算数值解的梯度
        bcs, ws = space.integrator.get_quadrature_points_and_weights()
        cellmeasure = space.cellmeasure 
        uh_grad_value = uh1.grad_value(bcs)**2  #(NQ,NC,2)
        uh_grad_value = np.sum(uh_grad_value,axis=2)
        uh_grad_norm = np.einsum('q,qc,c->', ws,uh_grad_value,cellmeasure)
        uh_grad_norm = np.sqrt(uh_grad_norm)
        
        vh_grad_value = vh1.grad_value(bcs)**2  #(NQ,NC,2)
        vh_grad_value = np.sum(vh_grad_value,axis=2)
        vh_grad_norm = np.einsum('q,qc,c->', ws,vh_grad_value,cellmeasure)
        vh_grad_norm = np.sqrt(vh_grad_norm)
        if (errU/uh_grad_norm < tol1) & (errV/vh_grad_norm < tol1):
            break
        else:
            #标记
            isMarkedCellU = smesh.refine_marker(etaU, theta, method='MAX')
            isMarkedCellV = smesh.refine_marker(etaV, theta, method='MAX')
            isMarkedCell = isMarkedCellU + isMarkedCellV

            #加密并将上一层的数值解插值到新网格上，同时保存网格图片
            NN0 = smesh.number_of_nodes()
            edge = smesh.entity('edge')
            smesh.refine_triangle_rg(isMarkedCell)
            refine_count += 1
            space = LagrangeFiniteElementSpace(smesh, p=1)
            print("t1",t1)
            print("refine_count",refine_count)
            print('refineerrU', errU)
            print('refinedof', space.number_of_global_dofs())
            print('ruh_grad_norm',uh_grad_norm)
            print('rerrU/uh_grad_norm ',errU/uh_grad_norm )
            print('refinedof', space.number_of_global_dofs())

            uh00 = space.function()
            vh00 = space.function()
            nn2e = smesh.newnode2edge
            uh00[:NN0] = uh0
            vh00[:NN0] = vh0
            uh00[NN0:] = np.average(uh0[edge[nn2e]], axis=-1)
            vh00[NN0:] = np.average(vh0[edge[nn2e]], axis=-1)
            uh0 = space.function()
            vh0 = space.function()
            uh0[:] = uh00
            vh0[:] = vh00
            if (abs(t1 - 0.1) < 1e-8) | (abs(t1 - 0.2) < 1e-8) | (abs(t1 - 0.3) < 1e-8) | (abs(t1 - 0.4) < 1e-8) :
                uh0.add_plot(plt,cmap='rainbow')
                plt.savefig('u/AFEM/uh-' + str(j) +'-t1-' + str(t1) +'-dof-' +          str(space.number_of_global_dofs()) + '.png')
                plt.close()
                vh0.add_plot(plt,cmap='rainbow')
                plt.savefig('v/AFEM/vh-' + str(j) + '-t1-'+ str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
                plt.close()
                smesh.add_plot(plt)
                plt.savefig('mesh/test-' + str(j) +  '-t1-' + str(t1) + '-dof-'+ str(space.number_of_global_dofs()) + '.png')
                plt.close()
                j += 1

    #粗化网格并插值
    isMarkedCellU = smesh.refine_marker(etaU, ctheta, 'COARSEN')
    isMarkedCellV = smesh.refine_marker(etaV, ctheta, 'COARSEN')
    isMarkedCell = isMarkedCellU & isMarkedCellV
    smesh.coarsen_triangle_rg(isMarkedCell)
    space = LagrangeFiniteElementSpace(smesh , p=1)
    print('coarsendof', space.number_of_global_dofs())
    uh2 = space.function()
    vh2 = space.function()
    retain = smesh.retainnode
    uh2[:] = uh1[retain]
    vh2[:] = vh1[retain]
    uh0 = space.function()
    vh0 = space.function()
    
    #为下一次计算作准备
    uh0[:] = uh2
    vh0[:] = vh2
    
    tmesh.advance()
plt.show()
