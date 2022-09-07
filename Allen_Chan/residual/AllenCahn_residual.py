#!/usr/bin/env python3
#
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
import argparse



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# 装饰子：指明被装饰函数输入的是笛卡尔坐标点
from fealpy.decorator import cartesian
import time
# 网格工厂：生成常用的简单区域上的网格
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh import HalfEdgeMesh2d

# 均匀剖分的时间离散
from fealpy.timeintegratoralg import UniformTimeLine

# AllenCahn_model
from AllenCahn_model_2d import TanhData as PDE

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

parser.add_argument('--TOLtime',
        default=0.05, type=float,
        help='时间误差限度，默认设定为 0.05.')

args = parser.parse_args()

degree = 1 #只能用p1元
ns = args.ns
nt = args.nt
tol = args.tol
T = args.T
tol0 = args.tol0
TOLtime = args.TOLtime
theta = 0.7
ctheta = 0.2
pde = PDE()
e = pde.e
domain = pde.domain()
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
def residual(uh0,uh1 , dt):
    space = uh0.space
    mesh = space.mesh
    hk = get_hk(mesh)
    measure = mesh.entity_measure('cell')
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()

    uhr0 = space.function()
    uhr1 = space.function()

    NC = mesh.number_of_cells()
    eta = np.zeros(NC, dtype=space.ftype)

    uvalue0 = uh0.value(bcs)
    uvalue1 = uh1.value(bcs)
    uvalue = (uvalue1 - uvalue0)/dt + (uvalue1**3 - uvalue1)/e**2
    
    #单元残量
    err_int = hk*np.sqrt(np.einsum('q,qc, qc, c -> c', ws , uvalue ,uvalue ,measure)) #(NC, )
    
    #边残量
    bc = np.array([1/3,1/3,1/3], dtype=space.ftype)
    grad = uh1.grad_value(bc)
    edge2cell = mesh.ds.edge_to_cell()
    n = mesh.edge_unit_normal()
    edgemeasure = mesh.entity_measure('edge')
    J = 0.5*edgemeasure**2*np.sum((grad[edge2cell[:, 0]] - grad[edge2cell[:,1]])*n, axis=-1)**2
    np.add.at(eta, edge2cell[:,0], J)
    np.add.at(eta, edge2cell[:,1], J)
    eta = np.sqrt(eta)
    eta = eta + err_int
    eta *= 1/16*0.25
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
    result = np.sqrt(space.integralalg.mesh_integral(f1, power=2))*Glw[0]
    result += np.sqrt(space.integralalg.mesh_integral(f2, power=2))*Glw[1]
    result += np.sqrt(space.integralalg.mesh_integral(f3, power=2))*Glw[2]
    result += np.sqrt(space.integralalg.mesh_integral(f4, power=2))*Glw[3]
    return result/(t1 - t0)

def velocity_matrix_test3( u, q=None):
    '''
    uh^2 \\cdot \phii \\cdot \\phij
    '''
    space = u.space
    mesh = space.mesh
    GD = mesh.geo_dimension()
    qf = space.integrator if q is None else mesh.integrator(q, etype='cell')
    # bcs.shape == (NQ, TD+1)
    # ws.shape == (NQ, )
    bcs, ws = qf.get_quadrature_points_and_weights()
    cellmeasure = space.cellmeasure
    gdof = space.number_of_global_dofs()
    c2d = space.cell_to_dof()
    uvalue = u.value(bcs)
    uvalue1 = uvalue**2
    phi = space.basis(bcs)
    shape = c2d.shape + c2d.shape[1:]
    I = np.broadcast_to(c2d[:, :, None], shape=shape)
    J = np.broadcast_to(c2d[:, None, :], shape=shape)
    val1 = np.einsum('q, qc, qci, qcj, c->cij', ws,
                    uvalue1, phi, phi, cellmeasure)
    M = csr_matrix(
        (val1.flat, (I.flat, J.flat)),
        shape=(gdof, gdof)
    )
    return M


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
time_step = []
log.write("time:" + str(0)+"--------------------------------------------------------------------------" + "\n" )
while True:
    # 初始网格的自适应
    space = LagrangeFiniteElementSpace(smesh, p=degree)
    # 当前时间步的有限元解
    uh0 = space.interpolation(pde.init_value)
    eta = residual(uh0,uh0, dt)
    err = np.sqrt(np.sum(eta**2))
    print("err", err)
    print("dofs:", space.number_of_global_dofs())


    log.write("count:" +  str(i)  + "            "  + "err:" + str(format(err,'.6f')) + "dof:" + str(space.number_of_global_dofs()) + "\n" )
    isMarkedCell = smesh.refine_marker(eta, theta, method='MAX')
    smesh.refine_triangle_rg(isMarkedCell)
    i += 1
    if err < tol:
         break


space = LagrangeFiniteElementSpace(smesh, p=degree)
uh0 = space.interpolation(pde.init_value)
uh00 = space.function()
uh00[:] = uh0
@cartesian
def source(p):
    return pde.source(p, t1)


t1 = 0
compute_time = 0
total_dof = 0
max_dof = space.number_of_global_dofs()
max_refine_count = 0
time_step_count = 0
start =time.time()
while t1 <= T:
    t1 = t0 + dt
    print("t1=", t1)

    log.write("time:" + str(t1)+"--------------------------------------------------------------------------" + "\n" )
    refine_count = 0
    time_step_count += 1
    time_step.append(dt)
    while True:
        uh1 = space.function()
        L = space.stiff_matrix()
        M = space.mass_matrix()

        Glp = (t1 - t0)/2*Gp + (t1 + t0)/2
        Glw = (t1 - t0)/2*Gw

        print("t0:",t0)
        print("t1:",t1)
        @cartesian
        def source_int(p):
            return (pde.source(p, Glp[0])*Glw[0] + pde.source(p, Glp[1])*Glw[1] + pde.source(p, Glp[2])*Glw[2] + pde.source(p, Glp[3])*Glw[3])/dt

        F = space.source_vector(source_int)
        #计算代数系统的右端项b
        b = 1/dt*M@uh0
        uh_old = space.function()
        uh_old[:] = uh0
        #牛顿迭代
        for j in range(20):
            J = velocity_matrix_test3(uh_old)
            #非线性方程求解矩阵
            A = M*1/dt + L  - e**(-2)*M + e**(-2)*J
            #Jacobi
            J = M*1/dt + L + 3*e**(-2)*J - e**(-2)*M
            #牛顿迭代
            res = b - A@uh_old
            uh1[:] = spsolve(J,res)
            uh1[:] += uh_old
            if np.linalg.norm((uh1[:] - uh_old[:])) < 10**(-5):
                print('-----------------')
                print('牛顿迭代次数:',j+1)
                print('norm:', np.linalg.norm((uh1[:] - uh_old[:])) )
                break
            uh_old[:] = uh1
        compute_time += 1
        total_dof += space.number_of_global_dofs()

        if(space.number_of_global_dofs() > max_dof):
            max_dof = space.number_of_global_dofs()
        eta = residual(uh0 ,uh1 ,t1 - t0)
        err = np.sqrt(np.sum(eta**2))
        eta_f = f_error_estimate(source_int,t0,t1)
        eta_time = time_error_estimate(uh0,uh1)


        energy_error.append(err)
        print("global error estimate:", err)

        if (eta_time > TOLtime/(2*T)) | (eta_f > np.sqrt(TOLtime)/(2*T)):
            print("eta_time:",eta_time)
            print("f_error:",eta_f)
            dt *= 0.5
            t1 = t0 + dt
            if t1 > T:
                dt = T - t0
                t1 = T
            continue

 
        if err< tol:
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
            uh_temp = space.function()
            nn2e = smesh.newnode2edge
            uh_temp[:NN0] = uh0
            uh_temp[NN0:] = np.average(uh0[edge[nn2e]], axis=-1)
            uh0 = space.function()
            uh0[:] = uh_temp

            uh_temp[:NN0] = uh00
            uh_temp[NN0:] = np.average(uh00[edge[nn2e]], axis=-1)
            uh00 = space.function()
            uh00[:] = uh_temp

            refine_count += 1
            number_of_nodes.append(space.number_of_global_dofs())
    if(max_refine_count < refine_count):
        max_refine_count =  refine_count

    #粗化网格并插值
    total_eta += dt*err
    ferror += eta_f*dt

    eta0 = residual(uh00 ,uh0 ,t1 - t0)
    isMarkedCell = smesh.refine_marker(eta, ctheta, 'COARSEN')
    isMarkedCell0 = smesh.refine_marker(eta0, ctheta, 'COARSEN')
    isMarkedCell = isMarkedCell & isMarkedCell0
    smesh.coarsen_triangle_rg(isMarkedCell)
    space = LagrangeFiniteElementSpace(smesh, p=degree)
    uh00 = space.function()
    retain = smesh.retainnode
    uh00[:] = uh0[retain]
    uh0 = space.function()
    uh0[:] = uh1[retain]


    number_of_nodes.append(space.number_of_global_dofs())

    if (eta_time <= 0.5*TOLtime/(2*T)) | (eta_f > np.sqrt(0.5*TOLtime)/(2*T)):
        dt *= 2
        tend = dt + t1
        if tend > T:
            dt = T - t1


    #画数值解图像
    if (abs(t1 - 1e-4) < 1e-5) | (abs(t1 - 0.049499999999999975) < 1e-5 ) | (abs(t1 -0.2015) < 1e-5) | (abs(t1 - 0.3006999999999999) < 1e-5) \
            | (abs(t1 - 0.4014999999999993) < 1e-4) | (abs(t1 - T) < 1e-4):
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1, projection='3d')
        #plt.title("T=" + str(t1) + ", dt = " + str(dt))
        uh0.add_plot(axes, cmap='rainbow')
        smesh.add_plot(plt)
        #plt.title("T=" + str(t1))
        plt.savefig('mesh/test-' + str(j) + "time:" + str(t1) + '-dof-'+ str(space.number_of_global_dofs()) +  '.png')
        plt.close()
    if t1 == T:
        break
    t0 = t1
end = time.time()
log.close()
eta_total = np.sqrt(total_eta +2*ferror**2)
print("eta:",eta_total)
print("average dof:", total_dof/compute_time)
print("max_dof:",max_dof)
print("max_refine_count:",max_refine_count)
print('Running time: %s Seconds'%(end-start))
plt.figure()
plt.tick_params(labelsize=18)
plt.xlabel("time",fontsize=24)
plt.ylabel("error estimate",fontsize=24)
time = np.linspace(0,T,len(energy_error))
plt.plot(time,energy_error)
plt.figure()
plt.tick_params(labelsize=18)
plt.xlabel("time",fontsize=24)
plt.ylabel("number of nodes",fontsize=24)
time = np.linspace(0,T,len(number_of_nodes))
plt.plot(time,number_of_nodes)
plt.figure()
plt.tick_params(labelsize=18)
plt.xlabel("time",fontsize=24)
plt.ylabel("time step",fontsize=24)
time = np.linspace(0,T,len(time_step))
plt.plot(time, time_step)
#show_error_table(NDof, errorType, errorMatrix, out='first.txt')
#showmultirate(plt, 0, NDof, errorMatrix, errorType, propsize=35)
plt.show()
