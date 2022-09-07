#!/usr/bin/env python3
#

from Keller_Segel_equation_2d import KSData2
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
theta = 0.8    
ctheta = 0.2

pde = KSData2()
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)
smesh.add_plot(plt)
plt.savefig('mesh/test' + str(0) + '.png')
plt.close()


tmesh = UniformTimeLine(0, tend, nt)
log = open("log.txt","w")
log.write("t1 = 0:"+ "\n" )
jj=1
j = 1
while True:
    space = LagrangeFiniteElementSpace(smesh, p=degree)
    uh0 = space.interpolation(pde.init_valueU)
    vh0 = space.interpolation(pde.init_valueV)
    etaU = space.recovery_estimate(uh0, method='area_harmonic')
    etaV = space.recovery_estimate(vh0, method='area_harmonic')
    errU = np.sqrt(np.sum(etaU**2))
    errV = np.sqrt(np.sum(etaV**2))
    #求数值解的梯度
    bcs, ws = space.integrator.get_quadrature_points_and_weights()
    cellmeasure = space.cellmeasure 
    uh_grad_value = uh0.grad_value(bcs)**2  #(NQ,NC,2)
    uh_grad_value = np.sum(uh_grad_value,axis=2)
    uh_grad_norm = np.einsum('q,qc,c->', ws,uh_grad_value,cellmeasure)
    uh_grad_norm = np.sqrt(uh_grad_norm)
    
    vh_grad_value = vh0.grad_value(bcs)**2  #(NQ,NC,2)
    vh_grad_value = np.sum(vh_grad_value,axis=2)
    vh_grad_norm = np.einsum('q,qc,c->', ws,vh_grad_value,cellmeasure)
    vh_grad_norm = np.sqrt(vh_grad_norm)
    
    '''
    rguh = space.grad_recovery(uh0)
    rgvh = space.grad_recovery(vh0)
    rguh_value = rguh.value(bcs)**2
    rguh_value = np.sum(rguh_value,axis=2)
    rguh_grad_norm = np.einsum('q,qc,c->', ws,rguh_value,cellmeasure)
    rguh_grad_norm = np.sqrt(rguh_grad_norm)

    rgvh_value = rgvh.value(bcs)**2
    rgvh_value = np.sum(rgvh_value,axis=2)
    rgvh_grad_norm = np.einsum('q,qc,c->', ws,rgvh_value,cellmeasure)
    rgvh_grad_norm = np.sqrt(rgvh_grad_norm)
    '''
    print('errU',errU)
    print('errV',errV)
    print('uh_grad_norm',uh_grad_norm)
    print('errU/uh_grad_norm ',errU/uh_grad_norm )
    print('errV/vh_grad_norm ',errV/vh_grad_norm )
    print('dof',space.number_of_global_dofs())
    isMarkedCellU = smesh.refine_marker(etaU, theta, method='MAX')
    isMarkedCellV = smesh.refine_marker(etaV, theta, method='MAX')
    isMarkedCell = isMarkedCellU + isMarkedCellV
    smesh.refine_triangle_rg(isMarkedCell)
    smesh.add_plot(plt)
    plt.savefig('mesh/test-' + str(jj) + '-dof-'+ str(space.number_of_global_dofs()) +  '.png')
    plt.close()
    jj += 1
    j += 1

    log.write("refine count:" + str(j) + "-"+ "dof:" + str(space.number_of_global_dofs()) + "\n" )
    #print("errU/rguh_grad_norm:",errU/rguh_grad_norm)
    #print("errV/rgvh_grad_norm:",errV/rgvh_grad_norm)
    if (errU/uh_grad_norm < tol) & (errV/vh_grad_norm < tol):
        break;
    #if space.number_of_global_dofs() > 10000:
     #   break;

dt = tmesh.current_time_step_length()

steps = np.arange(0,nt+1,1)
minU = []
minV = []
space = LagrangeFiniteElementSpace(smesh, p=degree)
uh0 = space.interpolation(pde.init_valueU)
vh0 = space.interpolation(pde.init_valueV)
uh0.add_plot(plt,cmap='rainbow')
plt.savefig('u/AFEM/uh-' + str(jj) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
plt.close()
vh0.add_plot(plt,cmap='rainbow')
plt.savefig('v/AFEM/vh-' + str(jj) + '-dof-' + str(space.number_of_global_dofs()) + '.png')
plt.close()
minU.append(min(uh0[:]))
minV.append(min(vh0[:]))
jj += 1
for k in range(nt):

    t1 = tmesh.next_time_level()
    print("t1:",t1)
    refine_count = 0
    error0=0
    errordown=0
    log.write("t1 = "+ str(t1) +":"  "\n" )
    j = 0
    while True:
        gdof = space.number_of_global_dofs()
        M = space.mass_matrix()
        L = space.stiff_matrix()

        #下一个时间层的解
        uh1 = space.function()
        vh1 = space.function()

        # 不动点迭代
        uh_new = space.function()
        vh_new = space.function()
        uh_old = space.function()
        vh_old = space.function()
        uh_old[:] = uh0
        vh_old[:] = vh0
        
        count = 0
        while True:
            AV = M + 0.5*dt*L + 0.5*dt*M
            bV = 0.5*dt*M@uh_old[:] + M@vh0[:] - 0.5*dt*L@vh0[:] + 0.5*dt*M@(uh0[:] - vh0[:])
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
            K2,K3 = space.ks_nonlinear_matrix(vh0)
            K2 += K3
            AU = M + 0.5*dt*L - 0.5*dt*K 
            bU = M@uh0[:] - 0.5*dt*L@uh0[:] + 0.5*dt*K2@uh0
            '''
            space.set_neumann_bc(F=bU, gN=neumann)
            AU, bU =bc.apply(bU, A = AU)
            '''
            uh_new[:] = spsolve(AU, bU)
            print("errorU",np.linalg.norm(uh_new - uh_old))
            print("errorV",np.linalg.norm(vh_new - vh_old))
            count += 1
            if(np.linalg.norm(uh_new - uh_old) < 1e-8) & (np.linalg.norm(vh_new - vh_old) < 1e-8):
                break
            if count > 50:
                break
            uh_old[:] = uh_new[:]
            vh_old[:] = vh_new[:]

        uh1[:] = uh_new[:]
        vh1[:] = vh_new[:]


        #计算误差估计子
        etaU = space.recovery_estimate(uh1, method='area_harmonic')
        etaV = space.recovery_estimate(vh1, method='area_harmonic')
        errU = np.sqrt(np.sum(etaU**2))
        errV = np.sqrt(np.sum(etaV**2))
        
        bcs, ws = space.integrator.get_quadrature_points_and_weights()
        cellmeasure = space.cellmeasure
        ''' 
        rguh = space.grad_recovery(uh1)
        rgvh = space.grad_recovery(vh1)
        rguh_value = rguh.value(bcs)**2
        rguh_value = np.sum(rguh_value,axis=2)
        rguh_grad_norm = np.einsum('q,qc,c->', ws,rguh_value,cellmeasure)
        rguh_grad_norm = np.sqrt(rguh_grad_norm)

        rgvh_value = rgvh.value(bcs)**2
        rgvh_value = np.sum(rgvh_value,axis=2)
        rgvh_grad_norm = np.einsum('q,qc,c->', ws,rgvh_value,cellmeasure)
        rgvh_grad_norm = np.sqrt(rgvh_grad_norm)          
        '''
        uh_grad_value = uh1.grad_value(bcs)**2  #(NQ,NC,2)
        uh_grad_value = np.sum(uh_grad_value,axis=2)
        uh_grad_norm = np.einsum('q,qc,c->', ws,uh_grad_value,cellmeasure)
        uh_grad_norm = np.sqrt(uh_grad_norm)
        
        vh_grad_value = vh1.grad_value(bcs)**2  #(NQ,NC,2)
        vh_grad_value = np.sum(vh_grad_value,axis=2)
        vh_grad_norm = np.einsum('q,qc,c->', ws,vh_grad_value,cellmeasure)
        vh_grad_norm = np.sqrt(vh_grad_norm)
        
    
        print("recovery_estimateU", errU)
        print("recovery_estimateV", errV)
        print("gradU", uh_grad_norm)
        print("gradV", vh_grad_norm)


        #print("errU/rguh_grad_norm:",errU/rguh_grad_norm)
        #print("errV/rgvh_grad_norm:",errV/rgvh_grad_norm)
        print("errU/uh_grad_norm:",errU/uh_grad_norm)
        print("errV/vh_grad_norm:",errV/vh_grad_norm)

        if (errU/uh_grad_norm < tol) & (errV/vh_grad_norm < tol):
            break
        #elif space.number_of_global_dofs() > 10000:
        #    break
        else:
            #标记
            isMarkedCellU = smesh.refine_marker(etaU, theta, method='MAX')
            isMarkedCellV = smesh.refine_marker(etaV, theta, method='MAX')
            isMarkedCell = isMarkedCellU + isMarkedCellV
            #加密并将上一层的数值解插值到新网格上，同时保存网格图片
            NN0 = smesh.number_of_nodes()
            edge = smesh.entity('edge')
            smesh.refine_triangle_rg(isMarkedCell)
            j += 1
           
            space = LagrangeFiniteElementSpace(smesh, p=1)
            log.write("refine count:" + str(j) + "-"+ "dof:" + str(space.number_of_global_dofs()) +"-" + "RelativeErrorEstimateU:" + str(errU/uh_grad_norm) + "-" +"RelativeErrorEstimateV:" + str(errV/vh_grad_norm)+ "\n" )
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


    #粗化网格并插值
    isMarkedCellU = smesh.refine_marker(etaU, ctheta, 'COARSEN')
    isMarkedCellV = smesh.refine_marker(etaV, ctheta, 'COARSEN')
    isMarkedCell = isMarkedCellU & isMarkedCellV
    smesh.coarsen_triangle_rg(isMarkedCell)
    space = LagrangeFiniteElementSpace(smesh , p=1)
    print('coarsendof', space.number_of_global_dofs())
    log.write("coarsendof dof:" + str(space.number_of_global_dofs()) + "\n" )
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

    if  (abs(t1 - tend)< 1e-8):
        smesh.add_plot(plt)
        plt.savefig('mesh/test-' + str(jj) +  '-t1-' + str(t1) + '-dof-'+ str(space.number_of_global_dofs()) + '.png')
        plt.close()
        uh0.add_plot(plt,cmap='rainbow')
        plt.savefig('u/AFEM/uh-' + str(jj) +'-t1-' + str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        vh0.add_plot(plt,cmap='rainbow')
        plt.savefig('v/AFEM/vh-' + str(jj) + '-t1-'+ str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        uh0.add_plot(axes, cmap='rainbow')
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        vh0.add_plot(axes, cmap='rainbow')
        
        minU.append(min(uh0[:]))
        minV.append(min(vh0[:]))
        jj += 1
    tmesh.advance()
log.close()
plt.show()
