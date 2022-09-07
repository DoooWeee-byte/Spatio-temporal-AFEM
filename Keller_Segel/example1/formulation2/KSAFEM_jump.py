#!/usr/bin/env python3
#
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
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
                    help='终止时刻，默认为 0.5 .')

parser.add_argument('--tol',
                    default=0.05, type=float,
                    help='容许误差，默认为 0.05 .')

args = parser.parse_args()

#设定参数
degree = 1
dim = args.dim
ns = args.ns
nt = args.nt
tend = args.T
tol = args.tol
theta = 0.8    
ctheta = 0.2

#导入模型
pde = KSData2()

#生成网格
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)

#时间一致剖分
tmesh = UniformTimeLine(0, tend, nt)

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

def jump_estimate(uh):
    space = uh.space
    mesh = space.mesh
    NC = mesh.number_of_cells()  # 单元的总个数
    eta = np.zeros(NC, dtype=space.ftype)
    bc = np.array([1/3,1/3,1/3], dtype=space.ftype) #重心坐标
    NC = mesh.number_of_cells()  #拿到单元的总个数
    grad = uh.grad_value(bc) #(NC, 2)计算每个单元的重心坐标处的梯度值
    edgemeasure = mesh.entity_measure('edge')  #每条边的长度
    n = mesh.edge_unit_normal() #(NE, 2) 每条边的外法向
    edge2cell = mesh.ds.edge_to_cell() #(NE, 4)
    J = 0.5*np.sum((grad[edge2cell[:,0]] - grad[edge2cell[:,1]])*n,axis=-1)**2*edgemeasure**2
    np.add.at(eta, edge2cell[:, 0], J)
    np.add.at(eta, edge2cell[:, 1], J)
    return np.sqrt(eta)


#新建一个文件用于记录网格和误差估计子的变化
log = open("log.txt","w")
log.write("time:" + str(0)+"--------------------------------------------------------------------------" + "\n" )

#记录初始加密次数
init_refine = 0

#这个循环是为了对初始网格进行加密
while True:
    #创建线性有限元空间
    space = LagrangeFiniteElementSpace(smesh, p=degree)
    
    #将初值函数插值到网格上
    uh0 = space.interpolation(pde.init_valueU)
    vh0 = space.interpolation(pde.init_valueV)
    
    #计算误差
    etaU = jump_estimate(uh0)
    etaV = jump_estimate(vh0)
    errU = np.sqrt(np.sum(etaU**2))
    errV = np.sqrt(np.sum(etaV**2))
    
    #求数值解的梯度
    bcs, ws = space.integrator.get_quadrature_points_and_weights()
    cellmeasure = space.cellmeasure 
    uh_grad_value = uh0.grad_value(bcs)**2  #(NQ,NC,2)
    uh_grad_value = np.sum(uh_grad_value,axis=2)
    uh_grad_norm = np.einsum('q,qc,c->', ws,uh_grad_value,cellmeasure)
    uh_grad_norm = np.sqrt(uh_grad_norm)  #uh的梯度
    
    vh_grad_value = vh0.grad_value(bcs)**2  #(NQ,NC,2)
    vh_grad_value = np.sum(vh_grad_value,axis=2)
    vh_grad_norm = np.einsum('q,qc,c->', ws,vh_grad_value,cellmeasure)
    vh_grad_norm = np.sqrt(vh_grad_norm)  #vh的梯度
    
    #打印出相对误差估计子和全局自由度,以便观察网格变化
    print('errV/vh_grad_norm ',errV/vh_grad_norm )
    print('errU/uh_grad_norm ',errU/uh_grad_norm )
    print('dof',space.number_of_global_dofs())
    
    #利用最大标记准则对单元进行标记
    isMarkedCellU = smesh.refine_marker(etaU, theta, method='MAX')
    isMarkedCellV = smesh.refine_marker(etaV, theta, method='MAX')
    isMarkedCell = isMarkedCellU | isMarkedCellV
    
    #红绿加密法对网格进行加密
    smesh.refine_triangle_rg(isMarkedCell)
    init_refine += 1
    
    #记录网格变化
    log.write("count:" +  str(init_refine)  + "            "  + "errorUh:" + str(format(errU,'.4f')) + "            "+ "errorVh:" + str(format(errV,'.4f')) + "            " + "dof:" + str(space.number_of_global_dofs()) + "\n" )
    print("errU", errU)
    print("errV", errV)
    #判断是否停止加密
    if (errV/vh_grad_norm< tol) &  (errU/uh_grad_norm < tol):
    #if errU/uh_grad_norm < tol:
        break;

#获得时间步长
dt = tmesh.current_time_step_length()

#创建空间
space = LagrangeFiniteElementSpace(smesh, p=degree)

#将初值函数插值到最新的网格上形成初始解向量
uh0 = space.interpolation(pde.init_valueU)
vh0 = space.interpolation(pde.init_valueV)

#时间迭代
for k in range(nt):
    
    #t1代表当前需要计算的时间层
    t1 = tmesh.next_time_level()
    
    #记录时间
    log.write("time:" + str(t1)+"--------------------------------------------------------------------------" + "\n" )
    
    #记录加密次数
    refine_count = 0
    
    #这个循环是为了对网格加密,得到满足迭出条件的网格
    while True:
        #M:质量矩阵   L:刚度矩阵  
        M = space.mass_matrix()
        L = space.stiff_matrix()
        
        #对M进行质量集中
        M = spdiags(np.sum(M, axis=1).flat,0,M.shape[0],M.shape[1])
        
        #下一个时间层的解
        uh1 = space.function()
        vh1 = space.function()

        #组装非线性项的矩阵 
        K,K1 = ks_nonlinear_matrix(vh0)
        K += K1
        
        #组装人工扩散矩阵
        D = -K.copy()
        D[(D -(D.transpose()))<0] = D[(D -(D.transpose()))<0]*0  + D.transpose()[(D -(D.transpose()))<0]
        D -= spdiags(D.diagonal(),0,K.shape[0],K.shape[1])
        D[D<0] = 0
        D -= spdiags(np.sum(D, axis=1).flat, 0, D.shape[0],D.shape[1])
        
        #合成左端矩阵
        A = M + dt/2*(L - K -D)

        #组装右端项
        b1 = (M - dt/2*(L - K - D))@uh0[:]

        #求解方程组
        uh1[:] = spsolve(A, b1)
        
        A1 = M + dt/2*L + dt/2*M
        F = dt/2*M@(uh1[:] + uh0[:])
        b2 = (M - dt/2*L - dt/2*M)@vh0[:] + F
        vh1[:] = spsolve(A1, b2)

        '''
        以下代码为利用普通的有限元格式计算,在较粗的网格上奇点附近会有振荡
        M = space.mass_matrix()
        L = space.stiff_matrix()
        
        K,K1 = space.ks_nonlinear_matrix(vh0)
        K += K1

        A = M + dt/2*(L - K)
        b1 = (M - dt/2*(L - K))@uh0[:]

        uh1[:] = spsolve(A, b1)
        A1 = M + dt/2*L + dt/2*M
        F = dt/2*M@(uh1[:] + uh0[:])
        b2 = (M - dt/2*L - dt/2*M)@vh0[:] + F
        vh1[:] = spsolve(A1, b2)
        '''
        
        #计算误差估计子
        etaU = jump_estimate(uh1) #局部误差估计子
        etaV = jump_estimate(vh1)
        errU = np.sqrt(np.sum(etaU**2))   #全局误差估计子
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

        print('errV/vh_grad_norm ',errV/vh_grad_norm )
        print('errU/uh_grad_norm ',errU/uh_grad_norm )
        print('dof',space.number_of_global_dofs())      
        #如果满足迭出条件就停止加密，否则进入标记和加密
        if (errV/vh_grad_norm< tol) &  (errU/uh_grad_norm < tol):
        #if errU/uh_grad_norm < tol:
            break
        else:
            #标记
            isMarkedCellU = smesh.refine_marker(etaU, theta, method='MAX')
            isMarkedCellV = smesh.refine_marker(etaV, theta, method='MAX')
            
            '''
            isMarkedCellU = etaU > (np.max(etaU)*0.8)
            isMarkedCellV = etaV > (np.max(etaV)*0.8)
            '''
            isMarkedCell = isMarkedCellU | isMarkedCellV
            
            #获取粗网格上的结点和边的信息方便之后的插值
            NN0 = smesh.number_of_nodes()
            edge = smesh.entity('edge')
            
            #红绿加密法对网格进行加密
            smesh.refine_triangle_rg(isMarkedCellU)

            #利用加密后的网格生成线性有限元空间
            space = LagrangeFiniteElementSpace(smesh, p=1)
            
            #打印相对误差和全局自由度,以便观察网格变化
            print('errV ',errV )
            print('errU ',errU )
            print('dof',space.number_of_global_dofs())
            
            #将uh0和vh0插值到新网格上
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
            
            #加密和插值完成后记录网格和误差估计子的变化
            refine_count += 1
            log.write("count:" +  str(refine_count)  + "            "  + "errorUh:" + str(format(errU,'.4f')) + "            "+ "errorVh:" + str(format(errV,'.4f')) + "            " + "dof:" + str(space.number_of_global_dofs()) + "\n" )

    #标记网格
    isMarkedCellU = smesh.refine_marker(etaU, ctheta, 'COARSEN')
    isMarkedCellV = smesh.refine_marker(etaV, ctheta, 'COARSEN')
    isMarkedCell = isMarkedCellU & isMarkedCellV
    
    #粗化网格
    smesh.coarsen_triangle_rg(isMarkedCellU)
    
    #利用粗化后的网格生成线性有限元空间
    space = LagrangeFiniteElementSpace(smesh , p=1)

    #将uh1和vh1插值到粗化后的网格上,为下一个时间层的计算做准备
    uh0 = space.function()
    vh0 = space.function()
    retain = smesh.retainnode
    uh0[:] = uh1[retain]
    vh0[:] = vh1[retain]
    log.write("coarsendof:"+ str(space.number_of_global_dofs()) + "\n" )
    
    #将最后一个时间层的网格和数值解保存并画图
    if  (abs(t1 - tend)< 1e-8):
        smesh.add_plot(plt)
        plt.savefig('mesh/test-' +  '-t1-' + str(t1) + '-dof-'+ str(space.number_of_global_dofs()) + '.png')
        plt.close()
        uh0.add_plot(plt,cmap='rainbow')
        plt.savefig('u/AFEM/uh-'+'-t1-' + str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        vh0.add_plot(plt,cmap='rainbow')
        plt.savefig('v/AFEM/vh-' + '-t1-'+ str(t1) +'-dof-' + str(space.number_of_global_dofs()) + '.png')
        plt.close()
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        uh0.add_plot(axes, cmap='rainbow')
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        vh0.add_plot(axes, cmap='rainbow')
    
    #时间层向前一层
    tmesh.advance()

#关闭文件,显示图像
log.close()
plt.show()
