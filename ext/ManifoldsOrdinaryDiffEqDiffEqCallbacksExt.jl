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
    solve_chart_parallel_transport_ode,
    solve_chart_volume_density
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

function chart_exp_problem!(du, u, params, t)
    M, A, i = params
    a = u.x[1]
    dx = u.x[2]
    copyto!(du.x[1], dx)
    affine_connection!(M, du.x[2], A, i, a, dx, dx)
    du.x[2] .*= -1
    return nothing
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
        prob = ODEProblem{true}(
            chart_exp_problem!, u0, (init_time, final_time), params; callback = cb
        )
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

function chart_pt_problem!(du, u, params, t)
    M, A, i = params
    a = u.x[1]
    dx = u.x[2]
    dY = u.x[3]

    copyto!(du.x[1], dx)
    affine_connection!(M, du.x[2], A, i, a, dx, dx)
    du.x[2] .*= -1
    affine_connection!(M, du.x[3], A, i, a, dx, dY)
    du.x[3] .*= -1
    return nothing
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
        prob = ODEProblem{true}(
            chart_pt_problem!, u0, (init_time, final_time), params; callback = cb
        )
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

function chart_jacobi_field_problem!(du, u, params, t)
    M, A, i = params
    a = u.x[1]
    dx = u.x[2]
    Y = u.x[3]
    dY = u.x[4]

    copyto!(du.x[1], dx)
    affine_connection!(M, du.x[2], A, i, a, dx, dx)
    du.x[2] .*= -1
    affine_connection!(M, du.x[4], A, i, a, dx, dY)
    du.x[4] .*= -1
    # temporarily save Riemann tensor value in du.x[3], then overwrite it with the final value later
    riemann_tensor!(M, du.x[3], A, i, a, Y, dx, dx)
    du.x[4] .-= du.x[3]
    affine_connection!(M, du.x[3], A, i, a, dx, Y)
    du.x[3] .*= -1
    du.x[3] .+= dY
    return nothing
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
        prob = ODEProblem{true}(
            chart_jacobi_field_problem!, u0, (init_time, final_time), params; callback = cb
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

function _jacobi_endpoint_coordinates(solution, final_time)
    sol, _ = solution.sols[end]
    return sol(final_time).x[3]
end

function _chart_jacobi_field_matrix_problem!(du, u, params, t)
    M, A, i = params
    a = u.x[1]
    dx = u.x[2]
    Y = u.x[3]
    dY = u.x[4]

    copyto!(du.x[1], dx)
    affine_connection!(M, du.x[2], A, i, a, dx, dx)
    du.x[2] .*= -1
    for j in axes(Y, 2)
        affine_connection!(M, view(du.x[4], :, j), A, i, a, dx, view(dY, :, j))
        view(du.x[4], :, j) .*= -1
        riemann_tensor!(M, view(du.x[3], :, j), A, i, a, view(Y, :, j), dx, dx)
        view(du.x[4], :, j) .-= view(du.x[3], :, j)
        affine_connection!(M, view(du.x[3], :, j), A, i, a, dx, view(Y, :, j))
        view(du.x[3], :, j) .*= -1
        view(du.x[3], :, j) .+= view(dY, :, j)
    end
    return nothing
end

function _transition_map_diff_matrix!(M, C_out, A, i_from, a, C_in, i_to)
    for j in axes(C_in, 2)
        transition_map_diff!(M, view(C_out, :, j), A, i_from, a, view(C_in, :, j), i_to)
    end
    return C_out
end

function _jacobi_exp_argument_matrix(
        M,
        a,
        Xc,
        A,
        i0;
        solver = AutoVern9(Rodas5P()),
        final_time::Real = 1.0,
        check_chart_switch_kwargs = NamedTuple(),
        kwargs...,
    )
    n = length(Xc)
    u0 = ArrayPartition(copy(a), copy(Xc), zeros(eltype(Xc), n, n), Matrix{eltype(Xc)}(I, n, n))
    cur_i = i0
    cb = FunctionCallingCallback(
        IntegratorTerminatorNearChartBoundary(check_chart_switch_kwargs);
        func_start = false,
    )
    retcode = SciMLBase.ReturnCode.Terminated
    init_time = zero(final_time)
    while retcode === SciMLBase.ReturnCode.Terminated && init_time < final_time
        params = (M, A, cur_i)
        prob = ODEProblem{true}(
            _chart_jacobi_field_matrix_problem!, u0, (init_time, final_time), params; callback = cb
        )
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]::typeof(final_time)
        a_final = sol.u[end].x[1]::typeof(a)
        new_i = get_chart_index(M, A, cur_i, a_final)
        if new_i !== cur_i
            transition_map!(M, u0.x[1], A, cur_i, new_i, a_final)
            transition_map_diff!(M, u0.x[2], A, cur_i, a_final, sol.u[end].x[2]::typeof(Xc), new_i)
            _transition_map_diff_matrix!(M, u0.x[3], A, cur_i, a_final, sol.u[end].x[3], new_i)
            _transition_map_diff_matrix!(M, u0.x[4], A, cur_i, a_final, sol.u[end].x[4], new_i)
            cur_i = new_i
        elseif retcode !== SciMLBase.ReturnCode.Terminated
            return sol.u[end].x[3], cur_i, a_final
        end
    end
    return u0.x[3], cur_i, u0.x[1]
end

raw"""
    solve_chart_volume_density(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0; kwargs...
    )

Compute the volume density of the exponential map in chart coordinates. The coordinates `a`
and `Xc` are represented in the induced basis of chart `i0` from atlas `A`.
"""
function solve_chart_volume_density(
        M::AbstractManifold, a, Xc, A::AbstractAtlas, i0; kwargs...
    )
    E, final_i, a_final = _jacobi_exp_argument_matrix(M, a, Xc, A, i0; kwargs...)
    return abs(det(E)) * sqrt(
        det_local_metric(M, A, final_i, a_final) / det_local_metric(M, A, i0, a)
    )
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
    E, _, _ = _jacobi_exp_argument_matrix(M, a, Xc, A, i0; kwargs...)
    final_time = get(kwargs, :final_time, 1.0)
    dYc = -E \ _jacobi_endpoint_coordinates(baseline, final_time)
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
    E, _, _ = _jacobi_exp_argument_matrix(M, a, Xc, A, i0; kwargs...)
    dYc = E \ Yc
    return solve_chart_jacobi_field(M, a, Xc, A, i0, zero(Yc), dYc; kwargs...)
end

end
