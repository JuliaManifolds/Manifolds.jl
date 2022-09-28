
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
        solver=MIRK(),
        kwargs...,
    )

Solve the BVP
"""
function solve_chart_log_bvp(
    M::AbstractManifold,
    a1,
    a2,
    A::AbstractAtlas,
    i;
    solver=GeneralMIRK4(),
    dt=0.05,
    kwargs...,
)
    tspan = (0.0, 1.0)
    function bc1!(residual, u, p, t)
        mid = div(length(u[1]), 2)
        residual[1:mid] = u[1][1:mid] - a1
        return residual[(mid + 1):end] = u[end][1:mid] - a2
    end
    bvp1 = BVProblem(
        chart_log_problem!,
        bc1!,
        (p, t) -> vcat(t * a1 + (1 - t) * a2, zero(a1)),
        tspan,
        (M, A, i),
    )
    sol1 = solve(bvp1, solver, dt=dt)
    return sol1
end

function estimate_distance_from_bvp(
    M::AbstractManifold,
    a1,
    a2,
    A::AbstractAtlas,
    i;
    solver=GeneralMIRK4(),
    dt=0.05,
    kwargs...,
)
    sol = solve_chart_log_bvp(M, a1, a2, A, i; solver, dt, kwargs...)
    mid = div(length(a1), 2)
    Xc = sol.u[1][1:mid]
    return norm(M, A, i, a1, Xc)
end
