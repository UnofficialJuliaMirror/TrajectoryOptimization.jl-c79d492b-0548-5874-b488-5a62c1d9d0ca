using RigidBodyDynamics
using ForwardDiff
using StaticArrays
using Test
using Random
using BenchmarkTools
Random.seed!(1)

traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf = joinpath(urdf_folder, "doublependulum.urdf")

mechanism = parse_urdf(urdf)

float64state = MechanismState(mechanism)
rand!(float64state)

q = configuration(float64state)
v = velocity(float64state)
momentum(float64state)

function momentum_vec(v::AbstractVector{T}) where T
    state = MechanismState{T}(mechanism)

    set_configuration!(state, q)
    set_velocity!(state, v)

    Vector(SVector(momentum(state)))
end

@test momentum_vec(v) == SVector(momentum(float64state))

J = ForwardDiff.jacobian(momentum_vec,v)

@benchmark ForwardDiff.jacobian($momentum_vec, $v)

# Speedup
const statecache = StateCache(mechanism)
const dynamicsresultscache = DynamicsResultCache(mechanism)

function momentum_vec!(out::AbstractVector, v::AbstractVector{T}) where T
    state = statecache[T]

    set_configuration!(state, q)
    set_velocity!(state, v)

    m = momentum(state)

    copyto!(out, SVector(m))
end

const out = zeros(6)
momentum_vec!(out,v)
@test out == SVector(momentum(float64state))

const result = DiffResults.JacobianResult(out,v)
const config = ForwardDiff.JacobianConfig(momentum_vec!, out, v)
ForwardDiff.jacobian!(result, momentum_vec!, out, v, config)
J = DiffResults.jacobian(result)

@benchmark ForwardDiff.jacobian!($result,$momentum_vec!,$out,$v,$config)

#############################################
# Improvement for TrajectoryOptimization.jl #
#############################################
traj_folder = joinpath(dirname(pathof(TrajectoryOptimization)),"..")
urdf_folder = joinpath(traj_folder, "dynamics/urdf")
urdf = joinpath(urdf_folder, "doublependulum.urdf")

model = Model(urdf)
n = model.n
m = model.m
ẋ1 = zeros(model.n)
ẋ2 = zeros(model.n)
x = rand(model.n)
u = rand(model.m)

torques = [1; 0]
statecache = StateCache(mechanism)
dynamicsresultscache = DynamicsResultCache(mechanism)
function f(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
    state = statecache[T]
    dyn = dynamicsresultscache[T]
    dynamics!(ẋ, dyn, state, x, u)
    return nothing
end

function f2(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
    state = statecache[T]
    dyn = dynamicsresultscache[T]
    q = x[1:2]
    q̇ = x[3:4]
    set_configuration!(state, q)
    set_velocity!(state, q̇)
    dynamics!(dyn, state, u)
    ẋ[1:2] = q̇
    ẋ[3:4] = dyn.v̇
    return nothing
end
dynamicsresult = DynamicsResult(mechanism)

f(ẋ1,x,u)
model.f(ẋ2,x,u)

@test isapprox(ẋ1,ẋ2)

@benchmark model.f($ẋ2,$x,$u)
@benchmark f($ẋ1,$x,$u)
@benchmark f2($ẋ1,$x,$u)


f_aug_old! = f_augmented!(model.f, n, m)
f_aug_fast! = f_augmented!(f2,n,m)

@test isapprox(ForwardDiff.jacobian(f_aug_old!,ẋ2,[x;u]), ForwardDiff.jacobian(f_aug_fast!,ẋ1,[x;u]))
