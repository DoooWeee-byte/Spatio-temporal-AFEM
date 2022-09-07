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

def interpolation(uhl1, uhl2, node1, node2):
    indices1 = []
    indices2 = []
    for i1 , row1 in enumerate(node1):
        for i2, row2 in enumerate(node2):
            if np.linalg.norm(row1-row2) < 1e-8:
                indices1.append(i1)
                indices2.append(i2)
    uhl2[indices2] = uhl1[indices1]
    size = len(indices2)
    while len(indices2) < len(node2):
        record = []
        for i, row in enumerate(node2):
            if np.isin(i,indices2):
                continue
            a = 100
            distance_min = 100
            for j in range(size-1):
                for k in range(j+1,size):
                    distance = np.linalg.norm(node2[j,:] - node2[k, :])
                    a = np.linalg.norm( (node2[j,:] + node2[k, :]) - 2*row )
                    if  (a < 1e-8) & (distance < distance_min):
                        uhl2[i] = (uhl2[j] + uhl2[k])/2
                        record.append(i)
                        distance_min = distance                           
        indices2 += record
        size += len(record)

#新建一个文件用于记录网格和误差估计子的变化
log = open("log.txt","w")
log.write("time:" + str(0)+"--------------------------------------------------------------------------" + "\n" )

#生成网格
domain = pde.domain()
smesh1 = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh1 = HalfEdgeMesh2d.from_mesh(smesh1, NV=3)




#记录初始加密次数
init_refine = 0

#这个循环是为了对初始网格进行加密
while True:
    #创建线性有限元空间
    space = LagrangeFiniteElementSpace(smesh1, p=degree)
    
    #将初值函数插值到网格上
    uh0 = space.interpolation(pde.init_valueU)
    vh0 = space.interpolation(pde.init_valueV)
    
    #计算误差
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
    isMarkedCellU = smesh1.refine_marker(etaU, theta, method='MAX')
    isMarkedCellV = smesh1.refine_marker(etaV, theta, method='MAX')
    isMarkedCell = isMarkedCellU | isMarkedCellV
    
    #红绿加密法对网格进行加密
    smesh1.refine_triangle_rg(isMarkedCell)
    init_refine += 1
    
    #记录网格变化
    log.write("count:" +  str(init_refine)  + "            "  + "errorUh:" + str(format(errU/uh_grad_norm,'.4f')) + "            "+ "errorVh:" + str(format(errV/vh_grad_norm,'.4f')) + "            " + "dof:" + str(space.number_of_global_dofs()) + "\n" )  
    #判断是否停止加密
    if (errU/uh_grad_norm < tol) &  (errV/vh_grad_norm < tol):
        break;
smesh1.add_plot(plt)
plt.savefig('mesh/test-' +  '-0-'+ '-dof-'+ str(space.number_of_global_dofs()) + '.png')
plt.close()

#获得时间步长
dt = tmesh.current_time_step_length()

#创建空间
space1 = LagrangeFiniteElementSpace(smesh1, p=degree)

#将初值函数插值到最新的网格上形成初始解向量
uh01 = space1.interpolation(pde.init_valueU)
vh01 = space1.interpolation(pde.init_valueV)
node1 = smesh1.node
#时间迭代
for k in range(nt):
    
    #t1代表当前需要计算的时间层
    t1 = tmesh.next_time_level()
    
    #记录时间
    log.write("time:" + str(t1)+"--------------------------------------------------------------------------" + "\n" )
    
    #记录加密次数
    refine_count = 0
    
    #生成第二个网格
    smesh2 = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
    smesh2 = HalfEdgeMesh2d.from_mesh(smesh2, NV=3)
    space2 = LagrangeFiniteElementSpace(smesh2, p=degree)
    uh02 = space2.function()
    vh02 = space2.function()
    node2 = smesh2.node
    interpolation(uh01, uh02, node1, node2)
    interpolation(vh01, vh02, node1, node2)
    
    #这个循环是为了对网格加密,得到满足迭出条件的网格
    while True:
        #M:质量矩阵   L:刚度矩阵  
        M = space2.mass_matrix()
        L = space2.stiff_matrix()
        
        #对M进行质量集中
        M = spdiags(np.sum(M, axis=1).flat,0,M.shape[0],M.shape[1])
        
        #下一个时间层的解
        uh12 = space2.function()
        vh12 = space2.function()

        #组装非线性项的矩阵 
        K,K1 = ks_nonlinear_matrix(vh02)
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
        b1 = (M - dt/2*(L - K - D))@uh02[:]

        #求解方程组
        uh12[:] = spsolve(A, b1)
        
        A1 = M + dt/2*L + dt/2*M
        F = dt/2*M@(uh12[:] + uh02[:])
        b2 = (M - dt/2*L - dt/2*M)@vh02[:] + F
        vh12[:] = spsolve(A1, b2)

        #计算误差估计子
        etaU = space2.recovery_estimate(uh12, method='area_harmonic') #局部误差估计子
        etaV = space2.recovery_estimate(vh12, method='area_harmonic')
        errU = np.sqrt(np.sum(etaU**2))   #全局误差估计子
        errV = np.sqrt(np.sum(etaV**2)) 
        
        #计算数值解的梯度
        bcs, ws = space2.integrator.get_quadrature_points_and_weights()
        cellmeasure = space2.cellmeasure 
        uh_grad_value = uh12.grad_value(bcs)**2  #(NQ,NC,2)
        uh_grad_value = np.sum(uh_grad_value,axis=2)
        uh_grad_norm = np.einsum('q,qc,c->', ws,uh_grad_value,cellmeasure)
        uh_grad_norm = np.sqrt(uh_grad_norm)
        
        vh_grad_value = vh12.grad_value(bcs)**2  #(NQ,NC,2)
        vh_grad_value = np.sum(vh_grad_value,axis=2)
        vh_grad_norm = np.einsum('q,qc,c->', ws,vh_grad_value,cellmeasure)
        vh_grad_norm = np.sqrt(vh_grad_norm)
        
        #打印相对误差和全局自由度,以便观察网格变化
        print('errV/vh_grad_norm ',errV/vh_grad_norm )
        print('errU/uh_grad_norm ',errU/uh_grad_norm )
        print('dof',space2.number_of_global_dofs())        
        
        #如果满足迭出条件就停止加密，否则进入标记和加密
        if (errU/uh_grad_norm < tol) & (errV/vh_grad_norm < tol):
            break
        else:
            #标记
            isMarkedCellU = smesh2.refine_marker(etaU, theta, method='MAX')
            isMarkedCellV = smesh2.refine_marker(etaV, theta, method='MAX')
            isMarkedCell = isMarkedCellU | isMarkedCellV
            
            #红绿加密法对网格进行加密
            smesh2.refine_triangle_rg(isMarkedCell)
            node2 = smesh2.node
            smesh2.add_plot(plt)
            plt.savefig('mesh/test-' +  '-t1-' + str(t1) + '-dof-'+ str(space2.number_of_global_dofs()) + '.png')
            plt.close()
            #利用加密后的网格生成线性有限元空间
            space2 = LagrangeFiniteElementSpace(smesh2, p=1)
            uh02 = space2.function()
            vh02 = space2.function()
            
            #把uh01,vh01插值到uh02,vh02上
            interpolation(uh01, uh02, node1, node2)
            interpolation(vh01, vh02, node1, node2)
                     
            #加密和插值完成后记录网格和误差估计子的变化
            refine_count += 1
            log.write("count:" +  str(refine_count)  + "            "  + "errorUh:" + str(format(errU/uh_grad_norm,'.4f')) + "            "+ "errorVh:" + str(format(errV/vh_grad_norm,'.4f')) + "            " + "dof:" + str(space2.number_of_global_dofs()) + "\n" )    
    
    #把smesh2赋值给smesh1
    smesh1 = copy.deepcopy(smesh2)
    node1 = smesh1.node
    #利用粗化后的网格生成线性有限元空间
    space1 = LagrangeFiniteElementSpace(smesh1 , p=1)
    uh01 = space1.function()
    vh01 = space1.function()
    uh01[:] = uh12
    vh01[:] = vh12
        
    #将最后一个时间层的网格和数值解保存并画图
    if  (abs(t1 - tend)< 1e-8):
        smesh1.add_plot(plt)
        plt.savefig('mesh/test-' +  '-t1-' + str(t1) + '-dof-'+ str(space1.number_of_global_dofs()) + '.png')
        plt.close()
        uh12.add_plot(plt,cmap='rainbow')
        plt.savefig('u/AFEM/uh-'+'-t1-' + str(t1) +'-dof-' + str(space1.number_of_global_dofs()) + '.png')
        plt.close()
        vh12.add_plot(plt,cmap='rainbow')
        plt.savefig('v/AFEM/vh-' + '-t1-'+ str(t1) +'-dof-' + str(space1.number_of_global_dofs()) + '.png')
        plt.close()
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        uh12.add_plot(axes, cmap='rainbow')
        
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        vh12.add_plot(axes, cmap='rainbow')
    
    #时间层向前一层
    tmesh.advance()

#关闭文件,显示图像
log.close()
plt.show()
