from dolfin import *
from mshr import *
import pandas as pd
import numpy as np
import os
import sys

# path to folder containing inclusions
incl_path = './inclusions/'

# path to location where results will be stored
rslt_path = './data/'

set_log_active(False)

#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 2

# Nonlinear optimizatin parameters
snes_Rtol         = 1e-9 # relative tolerance for SNES solver (phase field eq.)
snes_Atol         = 1e-9 # absolute tolerance for SNES solver (phase field eq.)
snes_maxiter      = 30   # max. iteration for SNEs solver (phase field eq.)

# Randomness
seed = np.random.randint(1e6)
np.random.seed(seed)

def grid_values(u, p_, res):
    """calculates field variables over a uniform grid with resolution "res" inside the domain

    Parameters
    ----------
    u : FEniCS Function
        displacements field
    p_ : FEniCS Function
        damage field
    res : int
        resolution of the grid

    Returns
    -------
        An (3,res,res) array containing [Xdisp, Ydisp, damage] over a uniform grid with resolution "res" 
    """
    disp_and_damage = np.zeros((3,res,res))
    for i in range(0,res):
        for j in range(0,res):
            xx = (j+0.5)/res - 0.5
            yy = (i+0.5)/res - 0.5
            disp_and_damage[:,i,j] = [u(xx,yy)[0], u(xx,yy)[1], p_(xx,yy)]
    return disp_and_damage


################################################
# Main
################################################
def main(case_num):
    """simulate phase-field method for sample "case_num" and record results

    Parameters
    ----------
    case_num : int
        sample number
    """
    # Creating the directory to store data
    newpath = rslt_path+'case' + str(case_num)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # constants
    P = 2
    lch = 0.095
    b = 0.015
    c_0 = 3.1415#8/3#2#3.1415
    a1 = 4*lch/(c_0*b)
    #a2 = -1/2
    #a3 = 0
    #ksi = 2
    l_o = b

    # Loading mesh
    mesh = Mesh('mesh.xml')

    # get top boundary nodes for Reaction force calculations
    def num_nem(seq, idfun=None):
        # order preserving
        if idfun is None:
            def idfun(x): return x
        seen = {}
        result = []
        for item in seq:
            marker = idfun(item)
            if marker in seen: continue
            seen[marker] = 1
            result.append(item)
        return result

    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.5)

    topBoundary = TopBoundary()
    ff = MeshFunction('size_t', mesh, 1)
    topBoundary.mark(ff,1)

    It_mesh = SubsetIterator(ff, 1)
    a = []
    bb = []
    for c in It_mesh:
        for v in vertices(c):
            a.append(v.midpoint().x())
            bb.append(v.midpoint().y())

    dd = num_nem(a)
    da = num_nem(bb)

    # Define Spaces, trial, and test funcitons
    V = FunctionSpace(mesh,'CG',1) # one dimensional space for damage field
    W = VectorFunctionSpace(mesh,'CG',1) # two dimensional space for displacement field
    VV = FunctionSpace(mesh, 'DG', 0) # one dimensional space for history variable

    # defining material properties
    fm = 4 # max rigidity ratio
    nu = 0.3

    with open(incl_path+'input' + str(case_num) + '.npy', 'rb') as file:
        centers = np.loadtxt(file)

    def bitmap(x,y):
        """calculates rigidity ratio at (x,y) in UFL

        Parameters
        ----------
        (x,y) : position of a point within the problem domain

        Returns
        -------
        rigidity ration at (x,y) in UFL
        """
        eta = 0.7 # shrinking ratio of Fashion MNIST bitmap
        d_min = (eta*0.3/28)**2 # radius of constant rigidity area around each inclusion center
        alpha = -100e1 # parameter of smooth min function
        beta = 0.9
        num = 0 # numerator of smooth min function
        denum = 0 # denominator of smooth min function
        for point in centers:
            px = eta*(point[0] / 28 - 0.5)
            py = eta*((28-point[1]) / 28 - 0.5)
            dist = (px-x)**2 + (py-y)**2
            num += dist*exp(alpha*dist)
            denum += exp(alpha*dist)
        d = num/denum
        return conditional(lt(d, d_min), 1, d_min/(beta*d_min + (1-beta)*d))

    XX = SpatialCoordinate(mesh)
    val = bitmap(XX[0], XX[1]) # variable rigidity ratio at (x,y): (val*(fm - 1) + 1)

    G_f = (val*(fm - 1) + 1) * 2.7
    f_t = (val*(fm - 1) + 1) * 2445.42
    E = (val*(fm - 1) + 1) * 210000.0

    lmbda, mu = (E*nu/((1.0 + nu )*(1.0-2.0*nu))) , (E/(2*(1+nu)))

    # Boundary conditions
    class right(SubDomain):
        def inside(self,x,on_boundary):
            return near(x[0], 0.5) and on_boundary

    class top(SubDomain):
        def inside(self,x,on_boundary):
            return near(x[1], 0.5) and on_boundary

    class bot(SubDomain):
        def inside(self,x,on_boundary):
            return near(x[1], -0.5) and on_boundary

    Right = right()
    Top = top()
    Bot = bot()

    u_topy = Expression("-t*(x[0] - 0.5)", t=0.0, degree=2)

    bc_rightx = DirichletBC(W.sub(0), Constant(0.0), Right)
    bc_topy = DirichletBC(W.sub(1), u_topy, Top)
    bc_boty = DirichletBC(W.sub(1), Constant(0.0), Bot)
    bc_botx = DirichletBC(W.sub(0), Constant(0.0), Bot)

    bc_u = [bc_rightx, bc_topy, bc_botx, bc_boty] # set of boundary conditions on the displacement field

    class crack(SubDomain):
        def inside(self,x,on_boundary):
            return abs(x[1]+0.2) <= 0.0002 and (x[0] <= -0.25) and on_boundary

    Crack = crack()

    bc_crack = DirichletBC(V, Constant(1.0), Crack)

    bc_phi = [bc_crack] # set of boundary conditions on the damage field

    # Define variational form
    p , q = TrialFunction(V), TestFunction(V)
    p_ = Function(V)
    u , v = TrialFunction(W), TestFunction(W)

    uold, unew, uconv1, umax = Function(W), Function(W), Function(W), Function(W)
    phiold, phiconv1, phimax = Function(V), Function(V), Function(V)

    VV = FunctionSpace(mesh, 'DG', 0)
    Hold, Hconv = Function(V), Function(V)

    def epsilon(u): # strain field
        return (grad(u) + grad(u).T) / 2

    def sigma_bar(eps):
        return 2*mu*eps + lmbda*tr(eps)*Identity(2)

    def omega(p):
        return (1-p)**2 / ((1-p)**2 + a1*p*(1-p/2)) # method 1
        return (1-p)**2 # method 2 & 3

    def sigma(eps, p):
        k = 1e-6
        return ((1-k)*omega(p) + k) * sigma_bar(eps)

    delta_u = inner(sigma(epsilon(u), phiold), epsilon(v)) * dx

    def sigma1(sigma_bar):
        sigma_x = sigma_bar[0,0]
        sigma_y = sigma_bar[1, 1]
        tau_xy = (sigma_bar[0, 1] + sigma_bar[1, 0]) / 2
        R = sqrt(tau_xy**2 + (sigma_x-sigma_y)**2 / 2)
        return (sigma_x+sigma_y)/2 + R

    def Y_bar(eps):
        #return 0.5*lmbda*tr(eps)**2 + mu*tr(eps*eps) # older method
        sigma = sigma1(sigma_bar(eps))
        return conditional(lt(sigma, 0), 0, sigma)**2 / (2*E)

    def d_omega(p):
        return a1*(p-1) / ((1-p)**2 + a1*p*(1-p/2))**2 # method 1
        return 2*(p-1) # method 2 & 3

    def hist(Y_bar, Hold):
        #return Y_bar # isotrpoic method
        TH = f_t**2 / (2*E)
        Y_bar = conditional(lt(Y_bar, TH), TH, Y_bar)
        return conditional(lt(Hold, Y_bar), Y_bar, Hold)

    def Y(p, Y_bar, Hold):
        return -d_omega(p)*hist(Y_bar, Hold)

    def d_alpha(p):
        return 2 - 2*p #method 1
        return 1 # method 2
        return 2*p # method 3

    def d_gamma(p, q):
        return (2*b*inner(grad(p), grad(q)) + d_alpha(p)*q/b) / c_0

    delta_phi = (-Y(p, Y_bar(epsilon(unew)), Hold)*q + G_f * d_gamma(p, q)) * dx
    delta_phi = action(delta_phi, p_)
    J_phi = derivative(delta_phi, p_, p)

    # Constraints for the phase field
    phi_min = interpolate(Constant(DOLFIN_EPS), V) # lower bound
    phi_max = interpolate(Constant(1.0), V)        # upper bound

    u = Function(W)
    p_disp = LinearVariationalProblem(lhs(delta_u), rhs(delta_u), u, bc_u)
    p_phi = NonlinearVariationalProblem(delta_phi, p_, bc_phi, J_phi)
    p_phi.set_bounds(phi_min, phi_max) # set bounds for the phase field

    # Construct solvers
    solver_disp = LinearVariationalSolver(p_disp)
    solver_phi  = NonlinearVariationalSolver(p_phi)

    snes_prm = {"nonlinear_solver": "snes",
                "snes_solver"     : { "method": "vinewtonssls",
                                    "line_search": "basic",
                                    "maximum_iterations": snes_maxiter,
                                    "relative_tolerance": snes_Rtol,
                                    "absolute_tolerance": snes_Atol,
                                    "report": False,
                                    "error_on_nonconvergence": False,
                                    }}
    solver_phi.parameters.update(snes_prm)

    # Solving variational problems
    forces = np.array([]).reshape(0,2)
    t = 0.0
    max_load = 0.02
    toll = 0.01
    B = 0 # vertical reaction force on top edge at each step
    fcounter = 0
    flag = 0
    flag10 = 0
    deltaT = 1e-4
    fmax = 0
    res = 256
    step_counter = 0
    info = np.zeros((20,3,res,res))
    while (t<= max_load):
        if flag == 1 and deltaT == 1e-4:
            t-=deltaT
            deltaT = 1e-5
            u.assign(uconv1)
            p_.assign(phiconv1)
            uold.assign(u)
            phiold.assign(p_)
            Hold.assign(Hconv)
        elif deltaT == 1e-5 and flag10 >= 10:
            deltaT = 1e-4
            flag10 = 0
        t+=deltaT
        print(f"t: {t}")
        u_topy.t = t
        iter = 1
        err = 1
        uconv1.assign(u)
        phiconv1.assign(p_)
        Hconv.assign(Hold)
        if forces.size and fmax < forces[-1][1]:
            fmax = forces[-1][1]
            umax.assign(u)
            phimax.assign(p_)
        while err > toll:
            if iter > 10 and deltaT == 1e-4:
                flag = 1
                break
            solver_disp.solve()
            unew.assign(u)
            Hold.assign(project(hist(Y_bar(epsilon(unew)), Hold), VV))
            solver_phi.solve()

            err_u = errornorm(u, uold, norm_type = 'l2', mesh = None)/norm(u)
            err_phi = errornorm(p_, phiold, norm_type = 'l2', mesh = None)/norm(p_)
            err = max(err_u, err_phi)
            print('iter', iter,'error', err)
            uold.assign(u)
            phiold.assign(p_)
            if err < toll:
                print('solution converges after:', iter)

                # calculate reaction force
                k_matx=assemble(delta_u)
                u_vec=u.vector()
                F_vec=k_matx*u_vec
                Ft_func=Function(W,F_vec)
                f_allY=[]
                for k in dd:
                    fy=Ft_func(k,0.5)
                    fyt=fy[1]
                    f_allY.append(fyt)
                    f_allY_s = f_allY
                    #f_y_u.append(f_allY_s)
                B=sum(f_allY_s)
                a = [(t,B)]
                forces = np.vstack((forces, [t, B*1e-3]))
                fcounter += 1
                flag = 0
                if deltaT == 1e-5:
                    flag10 += 1
            iter = iter+1
        if flag == 0 and np.round(t*100000)%100 == 0:
            info[step_counter] = grid_values(u, p_, res)
            step_counter += 1

    # save displacement and damage fields at every step
    ux = info[:,0,:,:]
    uy = info[:,1,:,:]
    damage = info[:,2,:,:]
    np.savetxt("./data/case" + str(case_num) + "/xdisp.npy", ux.reshape(20,256*256), fmt='%.2e')
    np.savetxt("./data/case" + str(case_num) + "/ydisp.npy", uy.reshape(20,256*256), fmt='%.2e')
    np.savetxt("./data/case" + str(case_num) + "/damage.npy", damage.reshape(20,256*256), fmt='%.2e')

    # Write u_final and phi_final to a file:
    addr = rslt_path+"case" + str(case_num)
    with HDF5File(MPI.comm_world, addr+"/displacements_final.h5", "w") as fFile:
        fFile.write(u,"data")
    with HDF5File(MPI.comm_world, addr+"/damage_final.h5", "w") as fFile:
        fFile.write(p_,"data")

    # Write u_max and phi_max to a file:
    with HDF5File(MPI.comm_world, addr+"/displacements_max_force.h5", "w") as fFile:
        fFile.write(umax,"data")
    with HDF5File(MPI.comm_world, addr+"/damage_max_force.h5", "w") as fFile:
        fFile.write(phimax,"data")

    with open(addr+"/forces"+"{:.4f}".format(l_o)+'.npy', 'wb') as file:
        np.save(file, forces)


if __name__ == '__main__':
    case_num = 0
    main(case_num)