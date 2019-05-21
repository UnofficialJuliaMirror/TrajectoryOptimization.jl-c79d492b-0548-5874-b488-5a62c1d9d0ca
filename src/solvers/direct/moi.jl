using MathOptInterface
const MOI = MathOptInterface

struct DirectProblem{T} <: MOI.AbstractNLPEvaluator
    prob::Problem{T,Continuous}
    solver::DIRCOLSolver{T,HermiteSimpson}
    jac_struct::Vector{NTuple{2,Int}}
    part_z::NamedTuple{(:X,:U), NTuple{2,Matrix{Int}}}
    p::NTuple{2,Int}    # (total constraints, p_colloc)
    nG::NTuple{2,Int}   # (total constraint jacobian, nG_colloc)
end

function DirectProblem(prob::Problem{T,Continuous}, solver::DIRCOLSolver{T,HermiteSimpson}) where T
    n,m,N = size(prob)
    p = num_constraints(prob)
    p_colloc = num_colloc(prob)
    P = p_colloc + sum(p)
    NN = N*(n+m)
    nG_colloc = p_colloc*2*(n + m)
    nG = nG_colloc + sum(p[1:N-1])*(n+m) + p[N]*n

    part_z = create_partition(n,m,N,N)

    jac_structure = spzeros(nG, NN)
    constraint_jacobian_sparsity!(jac_structure, prob)
    r,c = get_rc(jac_structure)
    jac_struct = collect(zip(r,c))
    num_con = (P,p_colloc)
    num_jac = (nG, nG_colloc)
    DirectProblem(prob, solver, jac_struct, part_z, num_con, num_jac)
end

MOI.features_available(d::DirectProblem) = [:Grad, :Jac]
MOI.initialize(d::DirectProblem, features) = nothing

MOI.jacobian_structure(d::DirectProblem) = d.jac_struct
MOI.hessian_lagrangian_structure(d::DirectProblem) = []

function MOI.eval_objective(d::DirectProblem, Z)
    X,U = unpack(Z, d.part_z)
    cost(d.prob.obj, X, U)
end

function MOI.eval_objective_gradient(d::DirectProblem, grad_f, Z)
    X,U = unpack(Z, d.part_z)
    cost_gradient!(grad_f, d.prob, X, U)
end

function MOI.eval_constraint(d::DirectProblem, g, Z)
    X,U = unpack(Z, d.part_z)
    P,p_colloc = d.p
    g_colloc = view(g, 1:p_colloc)
    g_custom = view(g, p_colloc+1:P)

    collocation_constraints!(g_colloc, d.prob, d.solver, X, U)
    update_constraints!(g_custom, d.prob, d.solver, X, U)
end

function MOI.eval_constraint_jacobian(d::DirectProblem, jac, Z)
    X,U = unpack(Z, d.part_z)
    n,m = size(d.prob)
    P,p_colloc = d.p
    nG_colloc = p_colloc * 2(n+m)
    jac_colloc = view(jac, 1:nG_colloc)
    collocation_constraint_jacobian!(jac_colloc, d.prob, d.solver, X, U)

    jac_custom = view(jac, nG_colloc+1:length(jac))
    constraint_jacobian!(jac_custom, d.prob, d.solver, X, U)
end

MOI.eval_hessian_lagrangian(::DirectProblem, H, x, σ, μ) = nothing