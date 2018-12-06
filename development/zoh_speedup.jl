using Test
using BenchmarkTools

##

model_pendulum, obj_uncon_pendulum = TrajectoryOptimization.Dynamics.pendulum!

u_min_pendulum = -2
u_max_pendulum = 2
x_min_pendulum = [-20;-20]
x_max_pendulum = [20; 20]

model_dubins, obj_uncon_dubins = TrajectoryOptimization.Dynamics.dubinscar!

# dubins car
u_min_dubins = [-1; -1]
u_max_dubins = [1; 1]
x_min_dubins = [0; -100; -100]
x_max_dubins = [1.0; 100; 100]
obj_con_dubins = ConstrainedObjective(obj_uncon_dubins, u_min=u_min_dubins, u_max=u_max_dubins, x_min=x_min_dubins, x_max=x_max_dubins)

# -Constrained objective
obj_con_pendulum = ConstrainedObjective(obj_uncon_pendulum, u_min=u_min_pendulum, u_max=u_max_pendulum, x_min=x_min_pendulum, x_max=x_max_pendulum)

model = model_dubins
obj = obj_con_dubins
u_max = u_max_dubins
u_min = u_min_dubins

# obj_min = TrajectoryOptimization.update_objective(obj, tf=:min, c=0.0, Q = 1e-3*Diagonal(I,model.n), R = 1e-3*Diagonal(I,model.m), Qf = Diagonal(I,model.n))

# Solver
intergrator = :rk3


dt = 0.005
solver1 = Solver(model,obj,integration=intergrator,N=51)
solver1.opts.minimum_time = false
solver1.opts.infeasible = false
solver1.opts.constrained = true
X0 = line_trajectory(solver1)
U0 = ones(solver1.model.m,solver1.N)
# U0 = [U0; ones(1,solver1.N)]
u0 = infeasible_controls(solver1,X0)
# U0 = [U0;u0]

solver2 = Solver(model,obj,integration=intergrator,N=51)
solver2.opts.minimum_time = false
solver2.opts.infeasible = false
solver2.opts.constrained = true

results1 = init_results(solver1,X0,U0)
results2 = init_results(solver2,X0,U0)
rollout!(results1,solver1)
rollout!(results2,solver2)
calculate_jacobians!(results1, solver1)
calculate_jacobians!(results2, solver2)

println("TEST")
@time v1 = _backwardpass_old!(results1,solver1)
@time v2 = _backwardpass!(results2,solver2)

@test isapprox(v1[1:2], v2[1:2])
@test isapprox(to_array(results1.K),to_array(results2.K))
@test isapprox(to_array(results1.b),to_array(results2.b))
@test isapprox(to_array(results1.d),to_array(results2.d))

# @btime _backwardpass_old!(results1,solver1)
# @btime _backwardpass!(results2,solver2)
println("\n")

# Benchmark
# 1. initial run
# 3.350 ms (3805 allocations: 341.98 KiB)
# 3.825 ms (4954 allocations: 410.63 KiB)


# model, obj = Dynamics.cartpole_analytical
# n,m = model.n, model.m
# N = 51
# dt = 0.1
#
# obj.x0 = [0;0;0;0]
# obj.xf = [0.5;pi;0;0]
# obj.tf = 2.0
# U0 = ones(m,N)
# solver_foh = Solver(model,obj,N=N,opts=opts,integration=:rk3_foh)
# solver_zoh = Solver(model,obj,N=N,opts=opts,integration=:rk3)
#
# k = 10
# time_per_iter_foh = zeros(k)
# time_per_iter_zoh = zeros(k)
#
# for i = 1:k
#   res_foh, stat_foh = solve(solver_foh,U0)
#   res_zoh, stat_zoh = solve(solver_zoh,U0)
#   time_per_iter_foh[i] = stat_foh["runtime"]/stat_foh["iterations"]
#   time_per_iter_zoh[i] = stat_zoh["runtime"]/stat_zoh["iterations"]
# end
# println("Time per iter (foh): $(mean(time_per_iter_foh))")
# println("Time per iter (zoh): $(mean(time_per_iter_zoh))")
