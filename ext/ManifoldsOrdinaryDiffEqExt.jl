module ManifoldsOrdinaryDiffEqExt

if isdefined(Base, :get_extension)
    using ManifoldsBase
    using ManifoldsBase: TraitList

    using Manifolds
    import Manifolds: exp!, solve_exp_ode
    using Manifolds: @einsum

    using ManifoldDiff: default_differential_backend

    using OrdinaryDiffEq: ODEProblem, AutoVern9, Rodas5, solve
    using StaticArrays
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..ManifoldsBase
    using ..ManifoldsBase: TraitList

    using ..Manifolds
    import ..Manifolds: exp!, solve_exp_ode
    using ..Manifolds: @einsum

    using ..ManifoldDiff: default_differential_backend

    using ..OrdinaryDiffEq: ODEProblem, AutoVern9, Rodas5, solve
    using ..StaticArrays
end

function solve_exp_ode(
    M::AbstractManifold,
    p,
    X,
    t::Number;
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    solver=AutoVern9(Rodas5()),
    backend=default_differential_backend(),
    kwargs...,
)
    n = length(p)
    iv = SVector{n}(1:n)
    ix = SVector{n}((n + 1):(2n))
    u0 = allocate(p, 2n)
    u0[iv] .= X
    u0[ix] .= p

    function exp_problem(u, params, t)
        M = params[1]
        dx = u[iv]
        p = u[ix]
        ddx = allocate(u, Size(n))
        du = allocate(u)
        Γ = christoffel_symbols_second(M, p, basis; backend=backend)
        @einsum ddx[k] = -Γ[k, i, j] * dx[i] * dx[j]
        du[iv] .= ddx
        du[ix] .= dx
        return Base.convert(typeof(u), du)
    end

    params = (M,)
    prob = ODEProblem(exp_problem, u0, (0.0, t), params)
    sol = solve(prob, solver; kwargs...)
    q = sol.u[end][(n + 1):(2 * n)]
    return q
end
# also define exp! for metric manifold anew in this case
function exp!(
    ::TraitList{IsMetricManifold},
    M::AbstractDecoratorManifold,
    q,
    p,
    X,
    t::Number;
    kwargs...,
)
    copyto!(M, q, solve_exp_ode(M, p, X, t; kwargs...))
    return q
end

end
