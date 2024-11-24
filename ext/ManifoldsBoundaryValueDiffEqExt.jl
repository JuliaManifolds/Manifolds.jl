module ManifoldsBoundaryValueDiffEqExt

using Manifolds
using ManifoldsBase

using Manifolds: affine_connection
import Manifolds: solve_chart_log_bvp, estimate_distance_from_bvp

using BoundaryValueDiffEq

function chart_log_problem!(du, u, params, t)
    M, A, i = params
    mid = div(length(u), 2)
    a = u[1:mid]
    dx = u[(mid + 1):end]
    ddx = -affine_connection(M, A, i, a, dx, dx)
    du[1:mid] .= dx
    du[(mid + 1):end] .= ddx
    return du
end

"""
    solve_chart_log_bvp(
        M::AbstractManifold,
        a1,
        a2,
        A::AbstractAtlas,
        i;
        solver=MIRK4(),
        dt::Real=0.05,
        kwargs...,
    )

Solve the BVP corresponding to geodesic calculation on [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) M,
between points with parameters `a1` and `a2` in a chart `i` of an [`AbstractAtlas`](@ref) `A`
using solver `solver`. Geodesic γ is sampled at time interval `dt`, with γ(0) = a1 and
γ(1) = a2.
"""
function solve_chart_log_bvp(
    M::AbstractManifold,
    a1,
    a2,
    A::AbstractAtlas,
    i;
    solver=MIRK4(),
    dt::Real=0.05,
    kwargs...,
)
    tspan = (0.0, 1.0)
    function bc1!(residual, u, p, t)
        mid = div(length(u[1]), 2)
        residual[1:mid] = u[1][1:mid] - a1
        residual[(mid + 1):end] = u[end][1:mid] - a2
        return residual
    end
    u0 = [vcat(a1, zero(a1)), vcat(a2, zero(a1))]
    bvp1 = BVProblem(chart_log_problem!, bc1!, u0, tspan, (M, A, i))
    sol1 = solve(bvp1, solver, dt=dt)
    return sol1
end

"""
    estimate_distance_from_bvp(
        M::AbstractManifold,
        a1,
        a2,
        A::AbstractAtlas,
        i;
        solver=MIRK4(),
        dt=0.05,
        kwargs...,
    )

Estimate distance between points on [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) M with parameters `a1` and
`a2` in chart `i` of [`AbstractAtlas`](@ref) `A` using solver `solver`, employing
[`solve_chart_log_bvp`](@ref) to solve the geodesic BVP.
"""
function estimate_distance_from_bvp(
    M::AbstractManifold,
    a1,
    a2,
    A::AbstractAtlas,
    i;
    solver=MIRK4(),
    dt=0.05,
    kwargs...,
)
    sol = solve_chart_log_bvp(M, a1, a2, A, i; solver, dt, kwargs...)
    mid = length(a1)
    Xc = sol.u[1][(mid + 1):end]
    return norm(M, A, i, a1, Xc)
end

end
