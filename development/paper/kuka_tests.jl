using MeshCatMechanisms
using LinearAlgebra
using RigidBodyDynamics
using Plots
import TrajectoryOptimization.rollout
include("N_plots.jl")
include("../kuka_visualizer.jl")
model, obj = Dynamics.kuka
n,m = model.n, model.m
nn = m   # Number of positions

function plot_sphere(frame::CartesianFrame3D,center,radius,mat,name="")
    geom = HyperSphere(Point3f0(center), convert(Float32,radius))
    setelement!(vis,frame,geom,mat,name)
end


# Create Mechanism
kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)
world = root_frame(kuka)

# Create Visualizer
visuals = URDFVisuals(Dynamics.urdf_kuka)
vis = MechanismVisualizer(kuka, visuals)
open(vis)


# Default solver options
opts = SolverOptions()
opts.cost_tolerance = 1e-6
opts.constraint_tolerance = 1e-5
dircol_options = Dict("tol"=>opts.cost_tolerance,"constr_viol_tol"=>opts.constraint_tolerance)


############################################
#    Unconstrained Joint Space Goal        #
############################################

# Define objective
x0 = zeros(n)
x0[2] = -pi/2
xf = copy(x0)
xf[1] = pi/4
Q = 1e-4*Diagonal(I,n)*10
Qf = 250.0*Diagonal(I,n)
R = 1e-5*Diagonal(I,m)/100
tf = 5.0
obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

# Define solver
N = 41
solver = Solver(model,obj_uncon,N=N, opts=opts)

# Generate initial control trajectory
U0_hold = hold_trajectory(solver, kuka, x0[1:7])
U0_hold[:,end] .= 0
X0_hold = TrajectoryOptimization.rollout(solver,U0_hold)

# Solve
solver.opts.verbose = true
solver.opts.live_plotting = false
solver.opts.iterations_innerloop = 200
solver.state.infeasible
solver.opts.bp_reg_initial = 0
res, stats = solve(solver,U0_hold)
stats["iterations"]
J = stats["cost"][end]
norm(res.X[N]-xf)

res_d, stats_d = solve_dircol(solver,X0_hold,U0_hold,options=dircol_options)
eval_f = gen_usrfun_ipopt(solver::Solver,:hermite_simpson)[1]
J - cost(solver,res_d.X,res_d.U)

set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res.X)
set_configuration!(vis, xf[1:7])
animate_trajectory(vis, res_d.X)
plot(to_array(res.U)')

plot(stats["cost"],yscale=:log10)
plot!(stats_d["cost"])

plot(res.U)

#####################################
#        WITH TORQUE LIMITS         #
#####################################

# Limit Torques
obj_con = ConstrainedObjective(obj_uncon, u_min=-75,u_max=75)
U_uncon = to_array(res.U)

solver = Solver(model,obj_con,N=N, opts=opts)
solver.opts.verbose = false
solver.opts.penalty_scaling = 50
solver.opts.penalty_initial = 0.0001
solver.opts.cost_tolerance_intermediate = 1e-3
solver.opts.outer_loop_update_type = :feedback
solver.opts.use_nesterov = false
solver.opts.iterations = 500
solver.opts.bp_reg_initial = 0

# iLQR
res_con, stats_con = solve(solver,U0_hold)
stats_con["iterations"]
cost(solver,res_con)

# DIRCOL
res_con_d, stats_con_d = solve_dircol(solver,X0_hold,U0_hold,options=dircol_options)
cost(solver,res_con_d)

p = convergence_plot(stats_con,stats_con_d,xscale=:log10)
TrajectoryOptimization.plot_vertical_lines!(p,stats_con["outer_updates"],title="Kuka Arm with Torque Limits")

# Visualize
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_con.X)

# Dircol Truth
group = "kuka/armswing/constrained"
solver_truth = Solver(model,obj_con,N=101)
run_dircol_truth(solver_truth, Array(res_con_d.X), Array(res_con_d.U), group::String)
X_truth, U_truth = get_dircol_truth(solver_truth,X0_hold,U0_hold,group)[2:3]
interp(t) = TrajectoryOptimization.interpolate_trajectory(solver_truth, X_truth, U_truth, t)

Ns = [41,51,75,101]
disable_logging(Logging.Debug)
run_step_size_comparison(model, obj_con, U0_hold, group, Ns, integrations=[:midpoint,:rk3],dt_truth=solver_truth.dt,opts=solver.opts,X0=X0_hold)
plot_stat("runtime",group,legend=:bottomright,title="Kuka Arm with Torque Limits")
plot_stat("iterations",group,legend=:bottom,title="Kuka Arm with Torque Limits")
plot_stat("error",group,yscale=:log10,legend=:left,title="Kuka Arm with Torque Limits")
plot_stat("c_max",group,yscale=:log10,legend=:bottom,title="Kuka Arm with Torque Limits")


####################################
#       END EFFECTOR GOAL          #
####################################

# Run ik to get final configuration to achieve goal
goal = [.5,.24,.9]
ik_res = Dynamics.kuka_ee_ik(kuka,goal)
xf[1:nn] = configuration(ik_res)

# Define objective with new final configuration
obj_ik = LQRObjective(Q, R, Qf, tf, x0, xf)

# Solve
solver_ik = Solver(model,obj_ik,N=N)
res_ik, stats_ik = solve(solver_ik,U0_hold)
stats_ik["iterations"]
cost(solver,res_ik)
norm(res_ik.X[N] - xf)
ee_ik = Dynamics.calc_ee_position(kuka,res_ik.X)
norm(ee_ik[N] - goal)

X0 = rollout(solver,U0_hold)
options = Dict("max_iter"=>10000,"tol"=>1e-6)
res_ik_d, stats_ik_d = solve_dircol(solver_ik,X0_hold,U0_hold,options=options)
cost(solver,res_ik_d)

# Plot the goal as a sphere
plot_sphere(world,goal,0.02,green_,"goal")
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_ik.X)


##########################################################
#          End-effector obstacle avoidance               #
##########################################################

# Define obstacles
r = 0.2
pos = Int.(floor.([0.12N, 0.28N]))
circles = [ee_ik[p] for p in pos]
for circle in circles; push!(circle,r) end
n_obstacles = length(circles)

import TrajectoryOptimization.sphere_constraint
state_cache = StateCache(kuka)
ee_body, ee_point = Dynamics.get_kuka_ee(kuka)
world = root_frame(kuka)

function cI(c,x::AbstractVector{T},u) where T
    state = state_cache[T]
    set_configuration!(state,x[1:nn])
    ee = transform(state,ee_point,world).v
    for i = 1:n_obstacles
        c[i] = sphere_constraint(ee,circles[i][1],circles[i][2],circles[i][3],r)
    end
end

function addcircles!(vis,circles)
    world = root_frame(kuka)
    for (i,circle) in enumerate(circles)
        p = Point3D(world,collect(circle[1:3]))
        setelement!(vis,p,circle[4],"obs$i")
    end
end

# Plot Obstacles in Visualizer
addcircles!(vis,circles)

# Formulate and solve problem
obj_obs = ConstrainedObjective(obj_ik,cI=cI)
solver = Solver(model, obj_obs, N=N)
solver.opts.verbose = false
solver.opts.penalty_scaling = 100
solver.opts.penalty_initial = 0.0001
# solver.opts.cost_tolerance = 1e-6
solver.opts.cost_tolerance_intermediate = 1e-3
# solver.opts.iterations = 200
solver.opts.bp_reg_initial = 10
U0 = to_array(res_ik.U)
U0_hold = hold_trajectory(solver,kuka,x0[1:nn])
res_obs, stats_obs = solve(solver,U0_hold)
cost(solver,res_obs)
X = res_obs.X
U = res_obs.U
ee_obs = Dynamics.calc_ee_position(kuka,X)

X0 = rollout(solver,U0_hold)
res_obs_d, stat_obs_d = solve_dircol(solver,X0,U0,options=options)
cost(solver,res_obs_d)

# Visualize
set_configuration!(vis, x0[1:7])
# animate_trajectory(vis, res_ik.X)
animate_trajectory(vis, res_obs.X, 0.2)

# Plot Constraints
C0 = zero.(res_obs.C)
for k = 1:N
    cI(C0[k],res_ik.X[k],res_ik.U[k])
end
cI(res_obs.C[N],X[N],U[N])
plot(to_array(C0)',legend=:bottom, width=2, color=[:blue :red :green :yellow])
plot!(to_array(res_obs.C)',legend=:bottom, color=[:blue :red :green :yellow])

# Compare EE trajectories
plot(to_array(ee_ik)',width=2)
plot!(to_array(ee_obs)')


##########################################################
#            Full body Obstacle Avoidance                #
##########################################################

# Collision Bubbles
function kuka_points(kuka,plot=false)
    bodies = [3,4,5,6]
    radii = [0.1,0.12,0.09,0.09]
    ee_body,ee_point = Dynamics.get_kuka_ee(kuka)
    ee_radii = 0.05

    points = Point3D[]
    frames = CartesianFrame3D[]
    for (idx,radius) in zip(bodies,radii)
        body = findbody(kuka,"iiwa_link_$idx")
        frame = default_frame(body)
        point = Point3D(frame,0.,0.,0.)
        plot ? plot_sphere(frame,0,radius,body_collision,"body_$idx") : nothing
        # plot ? setelement!(vis,point,radius,"body_$idx") : nothing
        push!(points,point)
        push!(frames,frame)
    end
    frame = default_frame(ee_body)
    plot ? plot_sphere(frame,0,ee_radii,body_collision,"end_effector") : nothing
    # plot ? setelement!(vis,ee_point,ee_radii,"end_effector") : nothing
    push!(points,ee_point)
    push!(radii,ee_radii)
    return points, radii, frames
end

function calc_kuka_points(x::Vector{T},points) where T
    state = state_cache[T]
    set_configuration!(state,x[1:nn])
    [transform(state,point,world).v for point in points]
end

function collision_constraint(c,obstacle,kuka_points,radii) where T
    for (i,kuka_point) in enumerate(kuka_points)
        c[i] = sphere_constraint(obstacle,kuka_point,radii[i]+obstacle[4])
    end
end

function generate_collision_constraint(kuka::Mechanism, circles)
    # Specify points along the arm
    points, radii = kuka_points(kuka)
    num_points = length(points)
    num_obstacles = length(circles)

    function cI_obstacles(c,x,u)
        nn = length(u)

        # Get current world location of points along arm
        arm_points = calc_kuka_points(x[1:nn],points)

        C = reshape(view(c,1:num_points*num_obstacles),num_points,num_obstacles)
        for i = 1:num_obstacles
            c_obstacle = view(C,1:num_points,i)
            collision_constraint(c_obstacle,circles[i],arm_points,radii)
        end

    end
end

# Add more obstacles
circles2 = copy(circles)
push!(circles2,[-0.3,0.5,0.7,0.2])
push!(circles2,[0.3,0.3,1.2,0.1])
push!(circles2,[0.3,-0.5,0.5,0.15])
addcircles!(vis,circles2)

points,radii,frames = kuka_points(kuka,true)


# Generate constraint function
c = zeros(length(points)*length(circles2))
cI_arm_obstacles = generate_collision_constraint(kuka,circles2)
cI_arm_obstacles(c,x0,zeros(m))

# Formulate and solve problem
costfun = LQRCost(Q,R,Qf,xf)
obj_obs_arm = ConstrainedObjective(obj_ik,cI=cI_arm_obstacles,u_min=-80,u_max=80)
obj_obs_arm.cost.Q = Q
obj_obs_arm.cost.R = R*1e-8

solver = Solver(model, obj_obs_arm, N=41)
solver.opts.verbose = true
solver.opts.penalty_scaling = 150
solver.opts.penalty_initial = 0.0005
# solver.opts.cost_tolerance = 1e-6
solver.opts.cost_tolerance_intermediate = 1e-2
solver.opts.cost_tolerance = 1e-7
# solver.opts.iterations = 200
solver.opts.bp_reg_initial = 0
solver.opts.outer_loop_update_type = :accelerated
res_obs, stats_obs = solve(solver,U0_hold)
X = res_obs.X
U = res_obs.U
ee_obs_arm = Dynamics.calc_ee_position(kuka,X)

U0_warm = to_array(U)

# Visualize
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_ik.X.x)
animate_trajectory(vis, res_obs.X.x, 0.2)

plot(to_array(res_obs.U)',legend=:none)

plot(to_array(res_obs.C)')

point_positions = [[zeros(3) for i = 1:length(points)] for k = 1:N]
for k = 1:N
    point_pos = calc_kuka_points(res_obs.X[k][1:nn],points)
    for i = 1:length(points)
        point_positions[k][i] = point_pos[i]
    end
end

dist = zeros(N)
for k = 1:N
    dist[k[]] = sphere_constraint(point_positions[k][2],circles2[3],circles2[3][4]+radii[2])
end
point_positions[2][1]

plot(dist)

c = zeros(5)
collision_constraint(c,circles2[3],point_positions[5],radii)
c

c = zeros(5*5)
cI_arm_obstacles(c,res_obs.X[5],res_obs.U[5])



##########################################################
#          End-effector frame Cost Funtion               #
##########################################################

function get_ee_costfun(mech,Qee,Q,R,Qf,ee_goal)
    statecache = StateCache(mech)
    state = MechanismState(mech)
    world = root_frame(mech)
    ee_pos = Dynamics.get_kuka_ee_postition_fun(mech,statecache)
    nn = num_positions(mech)
    n,m = 2nn,nn

    ee_body, ee_point = Dynamics.get_kuka_ee(kuka)
    p = path(mech, root_body(mech), ee_body)
    Jp = point_jacobian(state, p, transform(state, ee_point, world))

    # Hessian of the ee term wrt full state
    hess_ee = zeros(n,n)
    grad_ee = zeros(n)
    linds = LinearIndices(hess_ee)
    qq = linds[1:nn,1:nn]
    q_inds = 1:nn

    H = zeros(m,n)

    function costfun(x,u)
        ee = ee_pos(x) - ee_goal
        return (ee'Qee*ee + x'Q*x + u'R*u)/2
    end
    function costfun(xN::Vector{Float64})
        ee = ee_pos(xN) - ee_goal
        return 0.5*ee'Qf*ee + xN'Q*xN
    end

    function expansion(x::Vector{T},u) where T
        state = statecache[T]
        set_configuration!(state,x[1:nn])
        ee = ee_pos(x)

        point_in_world = transform(state, ee_point, world)
        point_jacobian!(Jp, state, p, point_in_world)
        Jv = Array(Jp)

        hess_ee[qq] = Jv'Qee*Jv
        Pxx = hess_ee + Q
        Puu = R
        Pux = H

        grad_ee[q_inds] = Jv'Qee*(ee-ee_goal)
        Px = grad_ee + Q*x
        Pu = R*u
        return Pxx,Puu,Pux,Px,Pu
    end
    function expansion(xN::AbstractVector{T}) where T
        state = statecache[T]
        set_configuration!(state,xN[1:nn])
        ee = ee_pos(xN)

        point_in_world = transform(state, ee_point, world)
        point_jacobian!(Jp, state, p, point_in_world)
        Jv = Array(Jp)

        hess_ee[qq] = Jv'Qf*Jv
        Pxx = hess_ee + Q

        grad_ee[q_inds] = Jv'Qf*(ee-ee_goal)
        Px = grad_ee + Q*xN
        return Pxx,Px
    end
    return costfun, expansion
end

Q2 = Diagonal([zeros(nn); ones(nn)*1e-5])
Qee = Diagonal(I,3)*1e-3
Qfee = Diagonal(I,3)*200
R2 = Diagonal(I,m)*1e-12

u0 = zeros(m)
eecostfun, expansion = get_ee_costfun(kuka,Qee,Q2,R2,Qfee,goal)
ee_costfun = GenericCost(eecostfun,eecostfun,expansion,n,m)
stage_cost(ee_costfun,x0,u0)
expansion(x0,u0)
expansion(x0)
TrajectoryOptimization.taylor_expansion(ee_costfun,x0)

ee_obj = UnconstrainedObjective(ee_costfun,tf,x0)
u_bnd = ones(m)*20
u_bnd[2] = 100
u_bnd[4] = 30
x_bnd = ones(n)*Inf
x_bnd[nn+1:end] .= 15
ee_obj_con = ConstrainedObjective(ee_obj,u_min=-u_bnd,u_max=u_bnd, x_min=-x_bnd, x_max=x_bnd,use_xf_equality_constraint=false)

U0_ik = to_array(res_ik.U)
solver_ee = Solver(model,ee_obj_con,N=61)
solver_ee.opts.cost_tolerance = 1e-6
solver_ee.opts.verbose = true
res,stats = solve(solver_ee,U0_hold)

set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res_ik.X)
animate_trajectory(vis, res.X)

plot(res.U,legend=:bottom)
qdot = to_array(res.X)[8:end,:]
plot(qdot')

res.C[N]
to_array(res.X)
res.X[end]
