#!/usr/bin/env python3
#

import argparse



import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import rc
rc('text', usetex=True)

# 装饰子：指明被装饰函数输入的是笛卡尔坐标点
from fealpy.decorator import cartesian

# 网格工厂：生成常用的简单区域上的网格
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh import HalfEdgeMesh2d

# 均匀剖分的时间离散
from fealpy.timeintegratoralg import UniformTimeLine

# 热传导 pde 模型
from heatequation_model_2d import ExpExpData

# Lagrange 有限元空间
from fealpy.functionspace import LagrangeFiniteElementSpace

# Dirichlet 边界条件
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table
#拷贝对象
import copy

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法求解热传导方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--ns',
        default=10, type=int,
        help='空间各个方向剖分段数， 默认剖分 10 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--tol',
        default=0.05, type=float,
        help='自适应加密停止限度，默认设定为 0.05 段.')

parser.add_argument('--tol0',
        default=1e-2, type=float,
        help='初始自适应加密停止限度，默认设定为 1e-3 段.')

parser.add_argument('--T',
        default=0.1, type=float,
        help='自适应加密停止限度，默认设定为 0.05 段.')

args = parser.parse_args()

degree = 1 #只能用p1元
ns = args.ns
nt = args.nt
tol = args.tol
T = args.T
tol0 = args.tol0
theta = 0.7
ctheta = 0.2
pde = ExpExpData()
domain = pde.domain()
c = pde.diffusionCoefficient

tmesh = UniformTimeLine(0, T, nt) # 均匀时间剖分

smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
smesh = HalfEdgeMesh2d.from_mesh(smesh, NV=3)
errorType = ['$\\eta_u$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$','$|| u -  u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((3, 5), dtype=np.float64)
NDof = np.zeros(5, dtype=np.int_)

#高斯点和权重，为了计算\int_{t^{n-1}}^{t^{n}} f(x,t) dt
Gw = np.array([0.3478548451,0.3478548451,0.6521451549,0.6521451549])
Gp = np.array([0.8611363116,-0.8611363116,0.3399810436,-0.3399810436])

#求出单元最大边长hk,形状是(NC,)
def get_hk(mesh):
    NC = mesh.number_of_cells()
    edgemeasure = mesh.entity_measure('edge')
    cell2edge = mesh.ds.cell_to_edge()
    a = np.vstack((edgemeasure[cell2edge[:,0]],edgemeasure[cell2edge[:,1]],edgemeasure[cell2edge[:,2]]))
    a = a.T
    a = np.max(a, axis=1)
    return a

#残量误差估计子
def residual(uh0,uh1,f, dt):
    space = uh0.space
    mesh = space.mesh
    measure = mesh.entity_measure('cell')
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
     
    #防止uh0和uh1被修改
    uhr0 = space.function()
    uhr1 = space.function()
    
    NC = mesh.number_of_cells()
    eta = np.zeros(NC, dtype=space.ftype)
    eta += space.integralalg.cell_integral(f, power=2)
    
    uh1_old = space.function()
    uh1_old[:] = uh1
    uhr1[:] = uh1 - uh0
    uhr0[:] = uhr1**2/dt**2
    uvalue0 = uhr0.value(bcs) #(NQ, NC)
    val = np.einsum('q, qc, c -> c', ws, uvalue0, measure)
    eta += val

    uvalue1 = uhr1.value(bcs)
    pp = mesh.bc_to_point(bcs)
    fval = f(pp) #(NQ, NC)
    fval = fval*2/dt
    val1 = np.einsum('q, qc ,qc, c -> c', ws, fval, uvalue1, measure)
    eta -= val1
    hk = get_hk(mesh)
    eta = 0.5*hk**2*eta

    bc = np.array([1/3,1/3,1/3], dtype=space.ftype)
    grad = uh1_old.grad_value(bc)
    edge2cell = mesh.ds.edge_to_cell()
    n = mesh.edge_unit_normal()
    edgemeasure = mesh.entity_measure('edge')
    J = edgemeasure**2*np.sum((grad[edge2cell[:, 0]] - grad[edge2cell[:,1]])*n, axis=-1)**2
    np.add.at(eta, edge2cell[:,0], J)
    np.add.at(eta, edge2cell[:,1], J)
    return eta

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
    return uh_grad_norm/3

#右端项的误差估计
def f_error_estimate(f_hat,t0,t1):
    Glp1 = (t1 - t0)/2*Gp + (t1 + t0)/2
    Glw1 = (t1 - t0)/2*Gw
    @cartesian
    def f1(p):
        return pde.source(p,Glp1[0]) - f_hat(p)
    @cartesian
    def f2(p):
        return pde.source(p,Glp1[1]) - f_hat(p)
    @cartesian
    def f3(p):
        return pde.source(p,Glp1[2]) - f_hat(p)
    @cartesian
    def f4(p):
        return pde.source(p,Glp1[3]) - f_hat(p)
    result = np.sqrt(space.integralalg.cell_integral(f1, power=2))*Glw[0]
    result += np.sqrt(space.integralalg.cell_integral(f2, power=2))*Glw[1]
    result += np.sqrt(space.integralalg.cell_integral(f3, power=2))*Glw[2]
    result += np.sqrt(space.integralalg.cell_integral(f4, power=2))*Glw[3]
    return result.sum()/(t1 - t0)
    
log = open("log.txt","w")
t0 = 0
dt = tmesh.current_time_step_length()
@cartesian
def source0(p):
    return pde.source(p,t0)

total_eta = 0  #total estimated error
epsilon = 0  #total energy error
ferror = 0
i = 0
energy_error = []
number_of_nodes = []
log.write("time:" + str(0)+"--------------------------------------------------------------------------" + "\n" )
while True:
    # 初始网格的自适应
    space = LagrangeFiniteElementSpace(smesh, p=degree)
    # 当前时间步的有限元解
    uh0 = space.interpolation(pde.init_value)
    eta = space.recovery_estimate(uh0, 'area_harmonic')
    err = np.sqrt(np.sum(eta**2))
    print("err", err)
    if err < tol0:
         break
    print("dofs:", space.number_of_global_dofs())
    log.write("count:" +  str(i)  + "            "  + "err:" + str(format(err,'.6f')) + "dof:" + str(space.number_of_global_dofs()) + "\n" )
    isMarkedCell = smesh.refine_marker(eta, theta, method='MAX')
    smesh.refine_triangle_rg(isMarkedCell)
    i += 1


space = LagrangeFiniteElementSpace(smesh, p=degree)
uh0 = space.interpolation(pde.init_value)

@cartesian
def source(p):
    return pde.source(p, t1)

@cartesian
def dirichlet(p):
    return pde.dirichlet(p, t1)

compute_time = 0
total_dof = 0
max_dof = space.number_of_global_dofs()
max_refine_count = 0
start =time.time()
for j in range(0, nt):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)
    t0 = t1 - dt
    Glp = (t1 - t0)/2*Gp + (t1 + t0)/2
    Glw = (t1 - t0)/2*Gw
    @cartesian
    def source_int(p):
        return (pde.source(p, Glp[0])*Glw[0] + pde.source(p, Glp[1])*Glw[1] + pde.source(p, Glp[2])*Glw[2] + pde.source(p, Glp[3])*Glw[3])/dt
    log.write("time:" + str(t1)+"--------------------------------------------------------------------------" + "\n" )
    refine_count = 0
    while True:
        uh1 = space.function()

        # if refine_count == 0:
        #uh0 = space.interpolation(pde.init_value)
        # 下一层时间步的有限元解
        A = c*space.stiff_matrix() # 刚度矩阵
        M = space.mass_matrix() # 质量矩阵
        dt = tmesh.current_time_step_length() # 时间步长
        G = M + dt*A # 隐式迭代矩阵
                    # t1 时间层的右端项

        F = space.source_vector(source_int)
        F *= dt
        F += M@uh0

        # t1 时间层的 Dirichlet 边界条件处理
        bc = DirichletBC(space, dirichlet)
        GD, F = bc.apply(G, F, uh1)

        # 代数系统求解
        uh1[:] = spsolve(GD, F)


        compute_time += 1
        total_dof += space.number_of_global_dofs() 
        if(space.number_of_global_dofs() > max_dof):
            max_dof = space.number_of_global_dofs()
        
        eta = space.recovery_estimate(uh1, 'area_harmonic')
        err = np.sqrt(np.sum(eta**2))
        eta_f = f_error_estimate(source_int,t0,t1)
        print('errrefine', err)
        
        # t1 时间层的误差
        @cartesian
        def solution(p):
            return pde.solution(p, t1)
        @cartesian
        def gradient(p):
            return pde.gradient(p, t1)
        L2error = space.integralalg.error(solution, uh1)
        H1error = space.integralalg.error(gradient, uh1.grad_value)
        energy_error.append(H1error)
        print("before interpolation L2:", L2error)
        print("before interpolation H1:", H1error)
        #print("time:",time_error_estimate(uh0,uh1))
        #print("f_error:",f_error_estimate(source_int,t0,t1))
        
        if err < tol:
            break
        else:
            #加密并插值
            NN0 = smesh.number_of_nodes()
            edge = smesh.entity('edge')
            isMarkedCell = smesh.refine_marker(eta, theta, method='MAX')
            smesh.refine_triangle_rg(isMarkedCell)
            i += 1
            space = LagrangeFiniteElementSpace(smesh, p=degree)
            print('refinedof', space.number_of_global_dofs())
            uh00 = space.function()
            nn2e = smesh.newnode2edge
            uh00[:NN0] = uh0
            uh00[NN0:] = np.average(uh0[edge[nn2e]], axis=-1)

            uh0 = space.function()
            uh0[:] = uh00
            print("refine_count:",refine_count)
            uh11 = space.function()
            uh11[:NN0] = uh1
            uh11[NN0:] = np.average(uh1[edge[nn2e]], axis=-1)
            refine_count += 1
            number_of_nodes.append(space.number_of_global_dofs())
            log.write("count:" +  str(refine_count)  + "            "  + "error:" + str(format(err,'.6f'))  + "            " +"errorL2:" + str(format(L2error,'.6f'))  + "            " +"errorH1:" + str(format(H1error,'.6f'))  + "            "  "dof:" + str(space.number_of_global_dofs()) + "\n" )
    if(max_refine_count < refine_count):
        max_refine_count =  refine_count  
       
    #粗化网格并插值
    total_eta += dt*err
    ferror += eta_f*dt
    epsilon += H1error**2*dt
    
    #eta0 = space.recovery_estimate(uh0, 'area_harmonic')
    #isMarkedCell0 = smesh.refine_marker(eta0, ctheta, 'COARSEN')
    isMarkedCell = smesh.refine_marker(eta, ctheta, 'COARSEN')
    #isMarkedCell = isMarkedCell0 & isMarkedCell
    smesh.coarsen_triangle_rg(isMarkedCell)
    space = LagrangeFiniteElementSpace(smesh, p=degree)
    uh0 = space.function()
    retain = smesh.retainnode
    uh0[:] = uh1[retain]
    log.write("coarsendof:"+ str(space.number_of_global_dofs()) + "\n" )
    
    
    number_of_nodes.append(space.number_of_global_dofs())
    #画数值解图像
    if (t1 ==0.01) | (t1 == 0.49) | (t1==0.99):
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        uh0.add_plot(axes, cmap='rainbow')
        smesh.add_plot(plt)
    
    # 时间步进一层
    tmesh.advance()
end = time.time()
log.close()
eta_total = np.sqrt(total_eta +2*ferror**2)
error_total_energy = np.sqrt(epsilon)
eff = eta_total/error_total_energy

print("eta:",eta_total)
print("error total_energy:",error_total_energy)
print("eff:" ,eff)
print("average dof:", total_dof/compute_time)
print("max_dof:",max_dof)
print("max_refine_count:",max_refine_count)
print('Running time: %s Seconds'%(end-start))

plt.figure()
plt.tick_params(labelsize=18)
plt.xlabel("time",fontsize=24)
plt.ylabel("energy error",fontsize=24)
time = np.linspace(0,1,len(energy_error))
plt.plot(time,energy_error)
plt.figure()
plt.tick_params(labelsize=18)
plt.xlabel("time",fontsize=24)
plt.ylabel("number of nodes",fontsize=24)
time = np.linspace(0,1,len(number_of_nodes))
plt.plot(time,number_of_nodes)
#show_error_table(NDof, errorType, errorMatrix, out='first.txt')
#showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=35)
plt.show()
