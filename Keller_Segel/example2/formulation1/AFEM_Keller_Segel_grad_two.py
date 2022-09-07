#!/usr/bin/env python3
#
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
from Keller_Segel_equation_2d import KSData10 as KSData
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
from fealpy.boundarycondition import NeumannBC 
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


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


def ks_nonlinear_matrix(uh,q=None):
    '''
    (u\\nabla v, \\phi)
    v是未知函数，\\phi是test function，u是已知函数
    '''
    space = uh.space
    mesh = space.mesh
    GD = mesh.geo_dimension()
    qf = space.integrator if q is None else mesh.integrator(q, etype='cell')
    # bcs.shape == (NQ, TD+1)
    # ws.shape == (NQ, )
    bcs, ws = qf.get_quadrature_points_and_weights()
    cellmeasure = space.cellmeasure
    gdof = space.number_of_global_dofs()
    c2d = space.cell_to_dof()
    ugrad = uh.grad_value(bcs)  # (NQ, NC, GD)
    # uvalue = uh.value(bcs)      #(NQ, NC)
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


args = parser.parse_args()

degree = 1
dim = args.dim
ns = args.ns
nt = args.nt
tend = args.T
tol = args.tol
tol1 = args.tol1
theta = 0.8   
ctheta = 0.3


pde = KSData()
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)
smesh.add_plot(plt)
plt.savefig('mesh/test' + str(0) + '.png')
plt.close()


def compute_t1(uh0,vh0,nnt):
    space = uh0.space
    uh1 = space.function()
    vh1 = space.function()
    M = space.mass_matrix()
    L = space.stiff_matrix()
    M = spdiags(np.sum(M, axis=1).flat,0,M.shape[0],M.shape[1])
    for jj in range(0, nnt):
        t = (jj + 1)*dt
        print("t:", t)
        
        #组装矩阵
        K,K1 = ks_nonlinear_matrix(vh0)
        K += K1
        D = -K.copy()
        D[(D -(D.transpose()))<0] = D[(D -(D.transpose()))<0]*0  + D.transpose()[(D -(D.transpose()))<0]
    
        D -= spdiags(D.diagonal(),0,K.shape[0],K.shape[1])
        D[D<0] = 0
        D -= spdiags(np.sum(D, axis=1).flat, 0, D.shape[0],D.shape[1])       
        A = M + dt/2*(L - K -D)
        
        #组装右端项
        b1 = (M - dt/2*(L - K - D))@uh0[:]

        #求解方程组
        uh1[:] = spsolve(A, b1)

        A1 = M + dt/2*L + dt/2*M
        F = dt/2*M@(uh1[:] + uh0[:])
        b2 = (M - dt/2*L - dt/2*M)@vh0[:] + F

        vh1[:] = spsolve(A1, b2)
        
        uh0[:] = uh1
        vh0[:] = vh1
    return [uh1,vh1]
      


tmesh = UniformTimeLine(0, tend, nt)
log = open("log.txt","w")
j = 1
init_refine = 0
log.write("time:" + str(0)+"--------------------------------------------------------------------------" + "\n" )
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
 
    #求数值解的梯度
    vh_grad_value = vh0.grad_value(bcs)**2  #(NQ,NC,2)
    vh_grad_value = np.sum(vh_grad_value,axis=2)
    vh_grad_norm = np.einsum('q,qc,c->', ws,vh_grad_value,cellmeasure)
    vh_grad_norm = np.sqrt(vh_grad_norm) 
    
    isMarkedCellU = smesh.refine_marker(etaU, theta, method='MAX')
    isMarkedCellV = smesh.refine_marker(etaV, theta, method='MAX')
    isMarkedCell = isMarkedCellU + isMarkedCellV
    smesh.refine_triangle_rg(isMarkedCell)
    init_refine += 1
    #log.write("init refine count:" + str(init_refine) +  "\n" )

    print("errU:",errU)
    print("errV:",errV)
    #print("errU/uh_grad_norm",errU/uh_grad_norm)
    #print("errV/vh_grad_norm",errV/vh_grad_norm)
    print(space.number_of_global_dofs())
    errV = errV/vh_grad_norm
    errU = errU/uh_grad_norm
    if(vh_grad_norm == 0):
        errV = 0
    log.write("count:" +  str(init_refine)  + "            "  + "errorUh:" + str(format(errU,'.4f')) + "            "+ "errorVh:" + str(format(errV,'.4f')) + "            " + "dof:" + str(space.number_of_global_dofs()) + "\n" )
    if (errU < tol) & (errV < tol):
        break

dt = tmesh.current_time_step_length()

space = LagrangeFiniteElementSpace(smesh, p=degree)
uh0 = space.interpolation(pde.init_valueU)
vh0 = space.interpolation(pde.init_valueV)



smesh.add_plot(plt)
plt.savefig('mesh/test-' + str(j) + '-dof-'+ str(space.number_of_global_dofs()) +  '.png')
plt.close()
for k in range(nt):
    t1 = tmesh.next_time_level()
    print("t1:",t1)
    log.write("time:" + str(t1)+"--------------------------------------------------------------------------" + "\n" )
    refine = False
    refine_count = 0
    while True:
        if (not refine):
            M = space.mass_matrix()
            L = space.stiff_matrix()
            M = spdiags(np.sum(M, axis=1).flat,0,M.shape[0],M.shape[1])

            #下一个时间层的解
            uh1 = space.function()
            vh1 = space.function()
            
            #组装矩阵
            K,K1 = ks_nonlinear_matrix(vh0)
            K += K1
            D = -K.copy()
            D[(D -(D.transpose()))<0] = D[(D -(D.transpose()))<0]*0  + D.transpose()[(D -(D.transpose()))<0]
    
            D -= spdiags(D.diagonal(),0,K.shape[0],K.shape[1])
            D[D<0] = 0
            D -= spdiags(np.sum(D, axis=1).flat, 0, D.shape[0],D.shape[1])            
            A = M + dt/2*(L - K -D)
            
            #组装右端项
            b1 = (M - dt/2*(L - K - D))@uh0[:]

            #求解方程组
            uh1[:] = spsolve(A, b1)

            A1 = M + dt/2*L + dt/2*M
            F = dt/2*M@(uh1[:] + uh0[:])
            b2 = (M - dt/2*L - dt/2*M)@vh0[:] + F

            vh1[:] = spsolve(A1, b2)

        else:
            uh0 = space.interpolation(pde.init_valueU)
            vh0 = space.interpolation(pde.init_valueV)
            uh1,vh1 = compute_t1(uh0,vh0,k+1)            
        

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
        print("errU:",errU)
        print("errV:",errV)      
        #print("errU/uh_grad_norm",errU/uh_grad_norm)
        #print("errV/vh_grad_norm",errV/vh_grad_norm)
        if (errU/uh_grad_norm < tol) & (errV/vh_grad_norm < tol):
            break
        else:
            #标记
            isMarkedCellU = smesh.refine_marker(etaU, theta, method='MAX')
            isMarkedCellV = smesh.refine_marker(etaV, theta, method='MAX')
            isMarkedCell = isMarkedCellU + isMarkedCellV
            #加密并将上一层的数值解插值到新网格上，同时保存网格图片
            smesh.refine_triangle_rg(isMarkedCell)
            refine = True
            refine_count += 1
            space=LagrangeFiniteElementSpace(smesh, p=degree)
            print('refinedof', space.number_of_global_dofs())
            log.write("count:" +  str(refine_count)  + "            "  + "errorUh:" + str(format(errU/uh_grad_norm,'.4f')) + "            "+ "errorVh:" + str(format(errV/vh_grad_norm,'.4f')) + "            " + "dof:" + str(space.number_of_global_dofs()) + "\n" )
            
 
    #粗化网格并插值
    isMarkedCellU = smesh.refine_marker(etaU, ctheta, 'COARSEN')
    isMarkedCellV = smesh.refine_marker(etaV, ctheta, 'COARSEN')
    isMarkedCell = isMarkedCellU & isMarkedCellV
    smesh.coarsen_triangle_rg(isMarkedCell)
    space = LagrangeFiniteElementSpace(smesh , p=1)
    print('coarsendof', space.number_of_global_dofs())
    uh0 = space.function()
    vh0 = space.function()
    retain = smesh.retainnode
    uh0[:] = uh1[retain]
    vh0[:] = vh1[retain]
    log.write("coarsendof:"+ str(space.number_of_global_dofs()) + "\n" )

    if  (abs(t1 - tend)< 1e-8):
        if (abs(t1 - tend)< 0.0001):
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1, projection='3d')
            uh0.add_plot(axes, cmap='rainbow')
        
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1, projection='3d')
            vh0.add_plot(axes, cmap='rainbow')
        smesh.add_plot(plt)
        plt.savefig('mesh/test-' + str(j) +  '-t1-' + str(t1) + '-dof-'+ str(space.number_of_global_dofs()) + '.png')
        plt.close()
        uh0.add_plot(plt,cmap='rainbow')
        plt.savefig('u/AFEM/uh-' + str(j) +'-t1-' + str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        vh0.add_plot(plt,cmap='rainbow')
        plt.savefig('v/AFEM/vh-' + str(j) + '-t1-'+ str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        
        j += 1
    tmesh.advance()

log.close()
plt.show()
