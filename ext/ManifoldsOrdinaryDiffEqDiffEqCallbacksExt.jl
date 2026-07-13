module ManifoldsOrdinaryDiffEqDiffEqCallbacksExt

using Manifolds
using Manifolds:
    IntegratorTerminatorNearChartBoundary,
    affine_connection,
    get_chart_index,
    riemann_tensor,
    transition_map!,
    transition_map_diff!
import Manifolds:
    solve_chart_differential_exp_argument,
    solve_chart_differential_exp_basepoint,
    solve_chart_differential_log_argument,
    solve_chart_differential_log_basepoint,
    solve_chart_exp_ode,
    solve_chart_jacobi_field,
    solve_chart_parallel_transport_ode
using ManifoldsBase

using DiffEqCallbacks
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqVerner: AutoVern9
using SciMLBase: SciMLBase, ODEProblem, solve

using LinearAlgebra
using RecursiveArrayTools: ArrayPartition

"""
    (int_term::IntegratorTerminatorNearChartBoundary)(u, t, integrator)

Terminate integration when integrator goes too closely to chart boundary.
Closeness is determined by `ϵ` value of [`IntegratorTerminatorNearChartBoundary`](@ref)
`int_term`.

# Arguments:

- `int_term`: object containing keyword arguments for `check_chart_switch`, such as
  the desired maximum distance to boundary,
- `u`: parameters of a point at which the integrator is solving a differential equation.
- `t`: time parameter of the integrator
- `integrator`: state of the integrator. Internal parameters are expected to contained
  the manifold on which the equation is solved, the atlas and the current chart index.
"""
function (int_term::IntegratorTerminatorNearChartBoundary)(u, t, integrator)
    (M, A, i) = integrator.p
    if check_chart_switch(M, A, i, u.x[1]; int_term.check_chart_switch_kwargs...)
        # switch charts
        SciMLBase.terminate!(integrator)
    end
    return u
end

@doc raw"""
    StitchedChartSolution{Prob,TM<:AbstractManifold,TA<:AbstractAtlas,TChart}

Solution of an ODE on a manifold `M` in charts of an [`AbstractAtlas`](@ref) `A`.

When `StitchedChartSolution{:Exp}` is used as a function with a number `t` as an argument, a
pair `(p, X)` is returned such that $p\in \mathcal{M}$ is the point at time `t` of the
geodesic and $X \in T_p \mathcal{M}$ is the velocity of the geodesic at that point.
Similarly, `StitchedChartSolution{:PT}` called with number `t` returns a triple `(p, X, Y)`
where `(p, X)` corresponds to the geodesic along which the vector is transported
and $Y\in T_p\mathcal{M}$ is the vector transported to `p`.
"""
struct StitchedChartSolution{Prob, TM <: AbstractManifold, TA <: AbstractAtlas, TChart}
    M::TM
    A::TA
    sols::Vector{Tuple{SciMLBase.AbstractODESolution, TChart}}
end

function StitchedChartSolution(
        M::AbstractManifold, A::AbstractAtlas, problem::Symbol, TChart
    )
    return StitchedChartSolution{problem, typeof(M), typeof(A), TChart}(M, A, [])
end

function (scs::StitchedChartSolution{:Exp})(t::Real)
    if t < scs.sols[1][1].t[1]
        throw(DomainError("Time $t is outside of the solution."))
    end
    for (sol, i) in scs.sols
        if t <= sol.t[end]
            B = induced_basis(scs.M, scs.A, i)
            solt = sol(t)
            p = get_point(scs.M, scs.A, i, solt.x[1])
            X = get_vector(scs.M, p, solt.x[2], B)
            return (p, X)
        end
    end
    throw(
        DomainError(
            "Time $t is outside of the solution (solution time range is [$(scs.sols[1][1].t[1]), $(scs.sols[end][1].t[end])]).",
        ),
    )
end
function (scs::StitchedChartSolution{:PT})(t::Real)
    if t < scs.sols[1][1].t[1]
        throw(DomainError("Time $t is outside of the solution."))
    end
    for (sol, i) in scs.sols
        if t <= sol.t[end]
            B = induced_basis(scs.M, scs.A, i)
            solt = sol(t)
            p = get_point(scs.M, scs.A, i, solt.x[1])
            X = get_vector(scs.M, p, solt.x[2], B)
            Y = get_vector(scs.M, p, solt.x[3], B)
            return (p, X, Y)
        end
    end
    throw(
        DomainError(
            "Time $t is outside of the solution (solution time range is [$(scs.sols[1][1].t[1]), $(scs.sols[end][1].t[end])]).",
        ),
    )
end
function (scs::StitchedChartSolution{:Jacobi})(t::Real)
    if t < scs.sols[1][1].t[1]
        throw(DomainError("Time $t is outside of the solution."))
    end
    for (sol, i) in scs.sols
        if t <= sol.t[end]
            B = induced_basis(scs.M, scs.A, i)
            solt = sol(t)
            p = get_point(scs.M, scs.A, i, solt.x[1])
            X = get_vector(scs.M, p, solt.x[2], B)
            Y = get_vector(scs.M, p, solt.x[3], B)
            dY = get_vector(scs.M, p, solt.x[4], B)
            return (p, X, Y, dY)
        end
    end
    throw(
        DomainError(
            "Time $t is outside of the solution (solution time range is [$(scs.sols[1][1].t[1]), $(scs.sols[end][1].t[end])]).",
        ),
    )
end

function (scs::StitchedChartSolution)(t::AbstractArray)
    return map(scs, t)
end

function chart_exp_problem(u, params, t)
    M, A, i = params
    a = u.x[1]
    dx = u.x[2]
    ddx = -affine_connection(M, A, i, a, dx, dx)
    return ArrayPartition(dx, ddx)
end

"""
    solve_chart_exp_ode(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0;
        solver=AutoVern9(Rodas5P()),
        final_time::Real=1.0,
        check_chart_switch_kwargs=NamedTuple(),
        kwargs...,
    )

Solve geodesic ODE on a manifold `M` from point of coordinates `a` in chart `i0` from an
[`AbstractAtlas`](@ref) `A` in direction of coordinates `Xc` in the induced basis.
The geodesic is solved up to time `final_time` (by default equal to 1).

## Chart switching

If the solution exceeds the domain of chart `i0` (which is detected using the
`check_chart_switch` function with additional keyword arguments `check_chart_switch_kwargs`),
a new chart is selected using `get_chart_index` on the final point in the old chart.

## Returned value

The function returns an object of type `StitchedChartSolution{:Exp}` to represent the
geodesic.
"""
function solve_chart_exp_ode(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0;
        solver = AutoVern9(Rodas5P()),
        final_time::Real = 1.0,
        check_chart_switch_kwargs = NamedTuple(),
        kwargs...,
    )
    u0 = ArrayPartition(copy(a), copy(Xc))
    cur_i = i0
    # callback stops solver when we get too close to chart boundary
    cb = FunctionCallingCallback(
        IntegratorTerminatorNearChartBoundary(check_chart_switch_kwargs); func_start = false
    )
    retcode = SciMLBase.ReturnCode.Terminated
    init_time = zero(final_time)
    sols = StitchedChartSolution(M, A, :Exp, typeof(i0))
    while retcode === SciMLBase.ReturnCode.Terminated && init_time < final_time
        params = (M, A, cur_i)
        prob =
            ODEProblem(chart_exp_problem, u0, (init_time, final_time), params; callback = cb)
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]::typeof(final_time)
        push!(sols.sols, (sol, cur_i))
        # here we switch charts
        a_final = sol.u[end].x[1]::typeof(a)
        new_i = get_chart_index(M, A, cur_i, a_final)
        if new_i !== cur_i
            transition_map!(M, u0.x[1], A, cur_i, new_i, a_final)
            transition_map_diff!(
                M, u0.x[2], A, cur_i, a_final, sol.u[end].x[2]::typeof(Xc), new_i
            )
            cur_i = new_i
        end
    end
    return sols
end

function chart_pt_problem(u, params, t)
    M, A, i = params
    a = u.x[1]
    dx = u.x[2]
    dY = u.x[3]

    ddx = -affine_connection(M, A, i, a, dx, dx)
    ddY = -affine_connection(M, A, i, a, dx, dY)
    return ArrayPartition(dx, ddx, ddY)
end

"""
    solve_chart_parallel_transport_ode(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc;
        solver=AutoVern9(Rodas5P()), check_chart_switch_kwargs=NamedTuple(), final_time=1.0,
        kwargs...
    )

Parallel transport vector with coordinates `Yc` along geodesic on a manifold `M` from point of
coordinates `a` in a chart `i0` from an [`AbstractAtlas`](@ref) `A` in direction of
coordinates `Xc` in the induced basis.
"""
function solve_chart_parallel_transport_ode(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc;
        solver = AutoVern9(Rodas5P()), final_time = 1.0, check_chart_switch_kwargs = NamedTuple(),
        kwargs...
    )
    u0 = ArrayPartition(copy(a), copy(Xc), copy(Yc))
    cur_i = i0
    # callback stops solver when we get too close to chart boundary
    cb = FunctionCallingCallback(
        IntegratorTerminatorNearChartBoundary(check_chart_switch_kwargs); func_start = false
    )
    retcode = SciMLBase.ReturnCode.Terminated
    init_time = zero(final_time)
    sols = StitchedChartSolution(M, A, :PT, typeof(i0))
    while retcode === SciMLBase.ReturnCode.Terminated && init_time < final_time
        params = (M, A, cur_i)
        prob =
            ODEProblem(chart_pt_problem, u0, (init_time, final_time), params; callback = cb)
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]::typeof(final_time)
        push!(sols.sols, (sol, cur_i))
        # here we switch charts
        a_final = sol.u[end].x[1]::typeof(a)
        new_i = get_chart_index(M, A, cur_i, a_final)
        transition_map!(M, u0.x[1], A, cur_i, new_i, a_final)
        transition_map_diff!(
            M, u0.x[2], A, cur_i, a_final, sol.u[end].x[2]::typeof(Xc), new_i
        )
        transition_map_diff!(
            M, u0.x[3], A, cur_i, a_final, sol.u[end].x[3]::typeof(Yc), new_i
        )
        cur_i = new_i
    end
    return sols
end

function chart_jacobi_field_problem(u, params, t)
    M, A, i = params
    a = u.x[1]
    dx = u.x[2]
    Y = u.x[3]
    dY = u.x[4]

    ddx = -affine_connection(M, A, i, a, dx, dx)
    dYdt = dY - affine_connection(M, A, i, a, dx, Y)
    ddY =
        -affine_connection(M, A, i, a, dx, dY) - riemann_tensor(M, A, i, a, Y, dx, dx)
    return ArrayPartition(dx, ddx, dYdt, ddY)
end

"""
    solve_chart_jacobi_field(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc, dYc;
        solver=AutoVern9(Rodas5P()), final_time=1.0,
        check_chart_switch_kwargs =NamedTuple(), kwargs...
    )

Solve the Jacobi equation along the geodesic starting at parameters `a` in chart `i0` with
initial velocity coordinates `Xc`. `Yc` and `dYc` are, respectively, the coordinates of the
initial Jacobi field and its initial covariant derivative in the induced basis of the chart.

The returned `StitchedChartSolution{:Jacobi}` returns `(p, X, Y, dY)` at time `t`, where `p`
is the point on the geodesic, `X` its velocity, `Y` the Jacobi field, and `dY` its covariant
derivative.
"""
function solve_chart_jacobi_field(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc, dYc;
        solver = AutoVern9(Rodas5P()), final_time::Real = 1.0,
        check_chart_switch_kwargs = NamedTuple(), kwargs...
    )
    u0 = ArrayPartition(copy(a), copy(Xc), copy(Yc), copy(dYc))
    cur_i = i0
    cb = FunctionCallingCallback(
        IntegratorTerminatorNearChartBoundary(check_chart_switch_kwargs); func_start = false
    )
    retcode = SciMLBase.ReturnCode.Terminated
    init_time = zero(final_time)
    sols = StitchedChartSolution(M, A, :Jacobi, typeof(i0))
    while retcode === SciMLBase.ReturnCode.Terminated && init_time < final_time
        params = (M, A, cur_i)
        prob = ODEProblem(
            chart_jacobi_field_problem, u0, (init_time, final_time), params; callback = cb
        )
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]::typeof(final_time)
        push!(sols.sols, (sol, cur_i))
        a_final = sol.u[end].x[1]::typeof(a)
        new_i = get_chart_index(M, A, cur_i, a_final)
        if new_i !== cur_i
            transition_map!(M, u0.x[1], A, cur_i, new_i, a_final)
            transition_map_diff!(
                M, u0.x[2], A, cur_i, a_final, sol.u[end].x[2]::typeof(Xc), new_i
            )
            transition_map_diff!(
                M, u0.x[3], A, cur_i, a_final, sol.u[end].x[3]::typeof(Yc), new_i
            )
            transition_map_diff!(
                M, u0.x[4], A, cur_i, a_final, sol.u[end].x[4]::typeof(dYc), new_i
            )
            cur_i = new_i
        end
    end
    return sols
end

function _jacobi_endpoint_coordinates(M, A, solution, final_time)
    p, _, Y, _ = solution(final_time)
    B = induced_basis(M, A, solution.sols[end][2])
    return get_coordinates(M, p, Y, B)
end

function _jacobi_exp_argument_matrix(M, a, Xc, A, i0, c; kwargs...)
    n = length(c)
    E = Matrix{eltype(c)}(undef, n, n)
    final_time = get(kwargs, :final_time, 1.0)
    for j in 1:n
        ej = zero(c)
        ej[j] = one(eltype(c))
        solution = solve_chart_jacobi_field(M, a, Xc, A, i0, zero(c), ej; kwargs...)
        E[:, j] .= _jacobi_endpoint_coordinates(M, A, solution, final_time)
    end
    return E
end

raw"""
    solve_chart_differential_exp_basepoint(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc; kwargs...
    )

Solve the Jacobi equation for ``D_p\exp_p(X)[Y]``. The coordinate vectors
`Xc` and `Yc` are represented in the chart-induced basis at `p`.
"""
function solve_chart_differential_exp_basepoint(
        M::AbstractManifold,
        a,
        Xc,
        A::AbstractAtlas,
        i0,
        Yc;
        kwargs...,
    )
    return solve_chart_jacobi_field(M, a, Xc, A, i0, Yc, zero(Yc); kwargs...)
end

raw"""
    solve_chart_differential_exp_argument(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc; kwargs...
    )

Solve the Jacobi equation for ``D_X\exp_p(X)[Y]``. The coordinate vectors
`Xc` and `Yc` are represented in the chart-induced basis at `p`.
"""
function solve_chart_differential_exp_argument(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc; kwargs...
    )
    return solve_chart_jacobi_field(M, a, Xc, A, i0, zero(Yc), Yc; kwargs...)
end

raw"""
    solve_chart_differential_log_basepoint(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc; kwargs...
    )

Solve the Jacobi equation for ``D_p\log_p(q)[Y]``, where
``q = \exp_p(X)``. The coordinate vector `Yc` is represented in the
chart-induced basis at `p`; the differential is the covariant derivative in
`solution(0)[4]`.
"""
function solve_chart_differential_log_basepoint(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc; kwargs...
    )
    baseline = solve_chart_jacobi_field(M, a, Xc, A, i0, Yc, zero(Yc); kwargs...)
    E = _jacobi_exp_argument_matrix(M, a, Xc, A, i0, Yc; kwargs...)
    final_time = get(kwargs, :final_time, 1.0)
    dYc = -E \ _jacobi_endpoint_coordinates(M, A, baseline, final_time)
    return solve_chart_jacobi_field(M, a, Xc, A, i0, Yc, dYc; kwargs...)
end

raw"""
    solve_chart_differential_log_argument(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc; kwargs...
    )

Solve the Jacobi equation for ``D_q\log_p(q)[Y]``, where
``q = \exp_p(X)``. The coordinate vector `Yc` is represented in the
chart-induced basis at `q`; the differential is the covariant derivative in
`solution(0)[4]`.
"""
function solve_chart_differential_log_argument(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0, Yc; kwargs...
    )
    E = _jacobi_exp_argument_matrix(M, a, Xc, A, i0, Yc; kwargs...)
    dYc = E \ Yc
    return solve_chart_jacobi_field(M, a, Xc, A, i0, zero(Yc), dYc; kwargs...)
end

end
