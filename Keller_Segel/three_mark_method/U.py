#!/usr/bin/env python3
#
from cProfile import label
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
from Keller_Segel_equation_2d import KSData2,KSData3
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
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

#拷贝对象
import copy

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
                    default=32, type=int,
                    help='空间各个方向剖分段数.')
parser.add_argument('--nt',
                    default=10, type=int,
                    help='时间剖分段数，默认剖分 100 段.')
parser.add_argument('--T',
                    default=2e-4, type=float,
                    help='终止时刻')
parser.add_argument('--tol',
                    default=0.25, type=float,
                    help='容许误差，默认为 0.05 .')
parser.add_argument('--theta',
                    default=0.7, type=float,
                    help='加密准则，默认设定为0.07.')
parser.add_argument('--ctheta',
                    default=0.2, type=float,
                    help='粗化准则，默认设定为0.2.')
parser.add_argument('--TOLcoarse',
                    default=1e-4, type=float,
                    help='粗化阈值，默认设定为0.01.')


args = parser.parse_args()

#设定参数
degree = 1
dim = args.dim
ns = args.ns
nt = args.nt
tend = args.T
T = tend
tol = args.tol
theta = args.theta    
ctheta = args.ctheta
TOLcoarse = args.TOLcoarse
#导入模型
pde = KSData3()

#生成网格
domain = pde.domain()
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)


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

def SCR(uh):
    # 样本点选取网格结点
    # 拿到需要用到的数据结构
    space = uh.space
    mesh = space.mesh
    node = mesh.entity("node")
    n2n = mesh.ds.node_to_node()
    is_bd = mesh.ds.boundary_node_flag()
    n2nI = n2n.indices
    n2nJ = n2n.indptr
    cRh = np.zeros((len(node), 3))
    bd_dic = {}
    for i in range(len(is_bd)):
        if is_bd[i] == False:  # 它是一个内部点
            connected_node = n2nI[n2nJ[i]:n2nJ[i+1]]
            A = node[connected_node]
            A = np.hstack([np.ones(len(A)).reshape(-1, 1), A])
            b = uh[connected_node]
            a = np.linalg.solve(A.transpose()@A, A.transpose() @ b)
            cRh[i] = a
        else:
            connect_node = n2nI[n2nJ[i]:n2nJ[i + 1]]
            for k in connect_node:
                if is_bd[k] == True:
                    continue
                else:
                    bd_dic[i] = [k]  # 第i号结点是一个边界点，它跟k号内部结点取相同的梯度值
    rguh = space.grad_recovery(uh, method="area_harmonic")
    rguh[~is_bd] = cRh[~is_bd][:, 1:]
    for key, value in bd_dic.items():
        rguh[key] = 0
        for x in value:
            rguh[key] += rguh[x]
        rguh[key] /= len(value)      
    eta = space.integralalg.error(rguh.value, uh.grad_value, power=2, celltype=True) # 计算单元上的恢复型误差
    return eta,rguh

#时间误差估计子
def time_error_estimate(uh0,uh1):
    space = uh0.space
    mesh = space.mesh
    cellmeasure = space.cellmeasure
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
    uht = space.function()
    uht[:] = uh1 - uh0
    ugrad = uht.grad_value(bcs)**2 #(NQ, NC, GD)
    uh_grad_value = np.sum(ugrad,axis=2)
    uh_grad_norm = np.einsum('q,qc,c->', ws,uh_grad_value,cellmeasure)
    return np.sqrt(uh_grad_norm)

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
log = open("U/log.txt","w")
log.write("time:" + str(0)+"--------------------------------------------------------------------------" + "\n")

#记录初始加密次数
init_refine = 0
time_list = []
dt_list = []
errU_list = []
dof_list = []
maxU_list = []
eta_list = []
time_list.append(0)
dt_list.append(T/nt)
maxU_list.append(420)
#这个循环是为了对初始网格进行加密
while True:
    #创建线性有限元空间
    space = LagrangeFiniteElementSpace(smesh, p=degree)
    
    #将初值函数插值到网格上
    uh0 = space.interpolation(pde.init_valueU)
    vh0 = space.interpolation(pde.init_valueV)
    
    #计算误差
    etaU = space.recovery_estimate(uh0, method='area_harmonic')
    etaV = space.recovery_estimate(vh0, method='area_harmonic')

    #etaU,_ = SCR(uh0)
    #etaV,_ = SCR(vh0)
    eta = etaU  
    
    # 全局误差估计子
    err = np.sqrt(np.sum(eta**2))
    
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
    
    grad_norm = np.sqrt(uh_grad_norm**2 + vh_grad_norm**2)
    
    #打印出相对误差估计子和全局自由度,以便观察网格变化
    print('err ',err )
    print('dof',space.number_of_global_dofs())
    
    #利用最大标记准则对单元进行标记
    isMarkedCell = smesh.refine_marker(eta, theta, method='MAX')
    
    #红绿加密法对网格进行加密
    smesh.refine_triangle_rg(isMarkedCell)
    init_refine += 1
    
    #记录网格变化
    #log.write("count:" +  str(init_refine)  + "            "  + "errorUh:" + str(format(errU/uh_grad_norm,'.4f')) + "            "+ "errorVh:" + str(format(errV/vh_grad_norm,'.4f')) + "            " + "dof:" + str(space.number_of_global_dofs()) + "\n" )
    
    #判断是否停止加密
    if (err/grad_norm < tol):
        break

eta_list.append(max(etaV)/max(etaU))
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
smesh.add_plot(ax)
plt.title(f"t = {0}", fontsize=24, loc='center',color='k')
plt.savefig('U/test-' +  '-t1-' + str(0) + '-dof-'+ str(space.number_of_global_dofs()) + '.pdf')
plt.close()
errU_list.append(err/grad_norm)


t0 = 0
dt = T/nt
tn = dt
tn_1 = 0
TOLtime = 30
TOLtime_min = 10
#创建空间
space = LagrangeFiniteElementSpace(smesh, p=degree)
dof_list.append(space.number_of_global_dofs())
#将初值函数插值到最新的网格上形成初始解向量
uh0 = space.interpolation(pde.init_valueU)
vh0 = space.interpolation(pde.init_valueV)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="z")
ax.tick_params(labelsize=16)
ax.zaxis.get_offset_text().set_fontsize(20)
plt.title(f"t = {0}", fontsize=24, loc='center',color='k')
uh0.add_plot(plt,cmap='rainbow')
plt.savefig('U/uh-'+'-t1-' + str(0) +'-dof-' + str(space.number_of_global_dofs()) + '.pdf')
plt.close()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="z")
ax.tick_params(labelsize=16)
ax.zaxis.get_offset_text().set_fontsize(20)
plt.title(f"t = {0}", fontsize=24, loc='center',color='k')
vh0.add_plot(plt,cmap='rainbow')
plt.savefig('U/vh-'+'-t1-' + str(0) +'-dof-' + str(space.number_of_global_dofs()) + '.pdf')
plt.close()


#时间迭代
while tn < T:
    tn = tn_1 + dt
    print("tn:", tn)
    
    #记录时间
    log.write("time:" + str(tn)+"--------------------------------------------------------------------------" + "\n" )
    #记录加密次数
    refine_count = 0
    
    #M:质量矩阵   L:刚度矩阵  
    M = space.mass_matrix()
    L = space.stiff_matrix()
    #下一个时间层的解
    uh1 = space.function()
    vh1 = space.function()
    #对M进行质量集中
    M = spdiags(np.sum(M, axis=1).flat,0,M.shape[0],M.shape[1])
    #组装非线性项的矩阵 
    K,K1 = ks_nonlinear_matrix(vh0)
    K += K1
    #组装人工扩散矩阵
    D = -K.copy()
    D[(D -(D.transpose()))<0] = D[(D -(D.transpose()))<0]*0  + D.transpose()[(D -(D.transpose()))<0]
    D -= spdiags(D.diagonal(),0,K.shape[0],K.shape[1])
    D[D<0] = 0
    D -= spdiags(np.sum(D, axis=1).flat, 0, D.shape[0],D.shape[1])
    #C-N格式
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
    # 对时间步长做自适应
    err_time = time_error_estimate(uh0, uh1)
    if (abs(tn - T) > 1e-10):
        if err_time > TOLtime:
            print("err_time", err_time)
            dt = 0.5*dt
            tn = tn_1
            continue
        elif err_time < TOLtime_min:
            print("err_time", err_time)
            dt = min(1.5*dt, T - tn_1)
            tn = tn_1
            continue
    
    #这个循环是为了对网格加密,得到满足迭出条件的网格
    while True:
        #M:质量矩阵   L:刚度矩阵  
        M = space.mass_matrix()
        L = space.stiff_matrix()

        #下一个时间层的解
        uh1 = space.function()
        vh1 = space.function()
        
        #对M进行质量集中
        M = spdiags(np.sum(M, axis=1).flat,0,M.shape[0],M.shape[1])
        
        #组装非线性项的矩阵 
        K,K1 = ks_nonlinear_matrix(vh0)
        K += K1
        
        #组装人工扩散矩阵
        D = -K.copy()
        D[(D -(D.transpose()))<0] = D[(D -(D.transpose()))<0]*0  + D.transpose()[(D -(D.transpose()))<0]
        D -= spdiags(D.diagonal(),0,K.shape[0],K.shape[1])
        D[D<0] = 0
        D -= spdiags(np.sum(D, axis=1).flat, 0, D.shape[0],D.shape[1])
            
        #C-N格式
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
         
        #计算误差估计子
        etaU = space.recovery_estimate(uh1, method='area_harmonic') #局部误差估计子
        etaV = space.recovery_estimate(vh1, method='area_harmonic')
        #etaU,_ = SCR(uh1)
        #etaV,_ = SCR(vh1)
        eta = etaU
        err = np.sqrt(np.sum(eta**2))   #全局误差估计子

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
        grad_norm = np.sqrt(uh_grad_norm **2 + vh_grad_norm **2)
        print("grad_norm:", grad_norm)
        print("err:", err)
        print("err/grad_norm", err/grad_norm)
          
        #如果满足迭出条件就停止加密，否则进入标记和加密
        if (err/grad_norm < tol):
            break
        else:
            #标记
            isMarkedCell = smesh.refine_marker(eta, theta, method='MAX')
            #获取粗网格上的结点和边的信息方便之后的插值
            NN0 = smesh.number_of_nodes()
            edge = smesh.entity('edge')  
            #红绿加密法对网格进行加密
            smesh.refine_triangle_rg(isMarkedCell)
            #利用加密后的网格生成线性有限元空间
            space = LagrangeFiniteElementSpace(smesh, p=1)
            
            #打印相对误差和全局自由度,以便观察网格变化
            print('err ', err)
            print('dof', space.number_of_global_dofs())
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
            log.write("count:" +  str(refine_count)  + "            "  + "errorUh:" + str(format(err/uh_grad_norm,'.4f')) + "            " + "dof:" + str(space.number_of_global_dofs()) + "\n" )

    #标记网格
    isMarkedCell = smesh.refine_marker(eta, ctheta, 'COARSEN')
    
    #粗化网格
    smesh.coarsen_triangle_rg(isMarkedCell)
    
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
    if  (abs(tn - tend)< 1e-15):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()
        smesh.add_plot(ax)
        plt.title(f"t = {tn}", fontsize=24, loc='center',color='k')
        plt.savefig('U/test-' +  '-t1-' + str(tn) + '-dof-'+ str(space.number_of_global_dofs()) + '.pdf')
        plt.close()
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="z")
        ax.tick_params(labelsize=16)
        ax.zaxis.get_offset_text().set_fontsize(20)
        plt.title(f"t = {tn}", fontsize=24, loc='center',color='k')
        uh0.add_plot(plt,cmap='rainbow')
        plt.savefig('U/uh-'+'-t1-' + str(tn) +'-dof-' + str(space.number_of_global_dofs()) + '.pdf')
        plt.close()

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="z")
        ax.tick_params(labelsize=16)
        ax.zaxis.get_offset_text().set_fontsize(20)
        plt.title(f"t = {tn}", fontsize=24, loc='center',color='k')
        vh0.add_plot(plt,cmap='rainbow')
        plt.savefig('U/vh-'+'-t1-' + str(tn) +'-dof-' + str(space.number_of_global_dofs()) + '.pdf')
        plt.close()
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot() 
        bcs = np.array([1/3, 1/3, 1/3])
        smesh.add_plot(plt,cellcolor=etaV,cmap=plt.cm.jet,linewidths=0,showaxis=True,showcolorbar=True)
        ax.tick_params(labelsize=16)
        plt.title(f"t = {tn},etaV", fontsize=24, loc='center',color='k')
        plt.savefig('eta_U/etaV-'+'-t1-' + str(tn) +'-dof-' + str(space.number_of_global_dofs()) + '.pdf')
        plt.close()

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot()
        bcs = np.array([1/3, 1/3, 1/3])
        smesh.add_plot(plt,cellcolor=etaU,cmap=plt.cm.jet,linewidths=0,showaxis=True,showcolorbar=True)
        ax.tick_params(labelsize=16)
        plt.title(f"t = {tn}, etaU", fontsize=24, loc='center',color='k')
        plt.savefig('eta_U/etaU-'+'-t1-' + str(tn) +'-dof-' + str(space.number_of_global_dofs()) + '.pdf')
        plt.close()


        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot()
        bcs = np.array([1/3, 1/3, 1/3])
        smesh.add_plot(plt,cellcolor=etaV/etaU,cmap=plt.cm.jet,linewidths=0,showaxis=True,showcolorbar=True)
        ax.tick_params(labelsize=16)
        plt.title(f"t = {tn},etaV/etaU", fontsize=24, loc='center',color='k')
        plt.savefig('eta_U/etaVU-'+'-t1-' + str(tn) +'-dof-' + str(space.number_of_global_dofs()) + '.pdf')
        plt.close()
        
        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot()
        # ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
        # ax.tick_params(labelsize=20)
        # plt.title(f"tn={tn},etaU", fontsize=24, loc='center',color='k')
        # ax.yaxis.get_offset_text().set_fontsize(20)
        # ax.plot(np.arange(len(etaU)), etaU, color='b', linewidth=1, linestyle='-')
        # plt.savefig('eta_U/lineU.pdf')
        # plt.close()

        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot()
        # ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
        # ax.tick_params(labelsize=20)
        # plt.title(f"tn={tn},etaV", fontsize=24, loc='center',color='k')
        # ax.yaxis.get_offset_text().set_fontsize(20)
        # ax.plot(np.arange(len(etaV)), etaV, color='b', linewidth=1, linestyle='-')
        # plt.savefig('eta_U/lineV.pdf')
        # plt.close()

        # fig = plt.figure(figsize=(8,6))
        # ax = fig.add_subplot()
        # ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
        # ax.tick_params(labelsize=20)
        # plt.title(f"tn={tn},etaV/etaU", fontsize=24, loc='center',color='k')
        # ax.yaxis.get_offset_text().set_fontsize(20)
        # ax.plot(np.arange(len(etaV)), etaV/etaU, color='b', linewidth=1, linestyle='-')
        # plt.savefig('eta_U/lineVU.pdf')
        # plt.close()
    
    time_list.append(tn)
    dt_list.append(dt)
    errU_list.append(err/grad_norm)
    dof_list.append(space.number_of_global_dofs())
    eta_list.append(max(etaV)/max(etaU))
    node = smesh.node
    maxU_list.append(uh0[~np.any(node, axis=1)][0])
    # 时间层前进一层
    tn_1 = tn
    dt = min(dt, T - tn_1)

print(len(time_list))  # nt + 1
print(len(dt_list))     # nt +1
print(len(errU_list))   # nt + 1
print(len(dof_list))     # nt + 1

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
ax.tick_params(labelsize=20)
plt.title("error estimate vs time", fontsize=24, loc='center',color='k')
ax.yaxis.get_offset_text().set_fontsize(20)
ax.plot(time_list, errU_list, color='b', linewidth=1, linestyle='-')
plt.savefig('U/error.pdf')
plt.close()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
ax.tick_params(labelsize=20)
plt.title("dt vs time", fontsize=24, loc='center',color='k')
ax.yaxis.get_offset_text().set_fontsize(20)
ax.plot(time_list, dt_list, color='b', linewidth=1, linestyle='-')
plt.savefig('U/dt.pdf')
plt.close()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
ax.tick_params(labelsize=20)
plt.title("dofs vs time", fontsize=24, loc='center',color='k')
ax.yaxis.get_offset_text().set_fontsize(20)
ax.plot(time_list, dof_list, color='b', linewidth=1, linestyle='-')
plt.savefig('U/dof.pdf')
plt.close()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
ax.tick_params(labelsize=20)
plt.title("max(Uh) vs time", fontsize=24, loc='center',color='k')
ax.yaxis.get_offset_text().set_fontsize(20)
ax.plot(time_list, maxU_list, color='b', linewidth=1, linestyle='-')
plt.savefig('U/maxU.pdf')
plt.close()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
ax.tick_params(labelsize=20)
plt.title("max(etaV/etaU) vs time", fontsize=24, loc='center',color='k')
ax.yaxis.get_offset_text().set_fontsize(20)
ax.plot(time_list, maxU_list, color='b', linewidth=1, linestyle='-')
plt.savefig('eta_U/VU.pdf')
plt.close()


#关闭文件,显示图像
log.close()

