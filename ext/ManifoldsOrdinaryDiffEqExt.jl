module ManifoldsOrdinaryDiffEqExt

using ManifoldsBase

using Manifolds
import Manifolds: exp!, exp_fused!, solve_exp_ode
using Manifolds: @einsum

using ManifoldDiff: default_differential_backend

using OrdinaryDiffEq: ODEProblem, AutoVern9, Rodas5, solve
using StaticArrays

function solve_exp_ode(
        M::AbstractManifold, p, X, t::Number;
        basis::AbstractBasis = DefaultOrthonormalBasis(),
        solver = AutoVern9(Rodas5()),
        backend = default_differential_backend(),
        kwargs...
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
        Γ = christoffel_symbols_second(M, p, basis; backend = backend)
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

function ManifoldsBase.retract_exp_ode!(
        M::AbstractManifold, q, p, X, m::AbstractRetractionMethod, b::AbstractBasis,
)
    ManifoldsBase.retract_exp_ode_fused!(M, q, p, X, one(number_eltype(p)), m, b)
    return q
end

function ManifoldsBase.retract_exp_ode_fused!(
    M::AbstractManifold,
    q,
    p,
    X,
    t::Number,
    ::AbstractRetractionMethod,
    b::AbstractBasis,
)
    sol = solve_exp_ode(M, p, X, t; basis=b)
    copyto!(q, sol)
    return q
end
end
