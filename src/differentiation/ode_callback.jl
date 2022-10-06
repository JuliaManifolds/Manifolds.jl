
"""
    IntegratorTerminatorNearChartBoundary{TKwargs}

An object for determining the point at which integration of a differential equation
in a chart on a manifold should be terminated for the purpose of switching a chart.

The value stored in `check_chart_switch_kwargs` will be passed as keyword arguments
to  [`check_chart_switch`](@ref). By default an empty tuple is stored.
"""
struct IntegratorTerminatorNearChartBoundary{TKwargs}
    check_chart_switch_kwargs::TKwargs
end

function IntegratorTerminatorNearChartBoundary()
    return IntegratorTerminatorNearChartBoundary(NamedTuple())
end

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
        OrdinaryDiffEq.terminate!(integrator)
    end
    return u
end

"""
    StitchedChartSolution{TM<:AbstractManifold,TA<:AbstractAtlas,TChart}

Solution of an ODE on a manifold `M` in charts of an [`AbstractAtlas`](@ref) `A`.
"""
struct StitchedChartSolution{Prob,TM<:AbstractManifold,TA<:AbstractAtlas,TChart}
    M::TM
    A::TA
    sols::Vector{Tuple{OrdinaryDiffEq.SciMLBase.AbstractODESolution,TChart}}
end

function StitchedChartSolution(
    M::AbstractManifold,
    A::AbstractAtlas,
    problem::Symbol,
    TChart,
)
    return StitchedChartSolution{problem,typeof(M),typeof(A),TChart}(M, A, [])
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
    throw(DomainError("Time $t is outside of the solution."))
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
    throw(DomainError("Time $t is outside of the solution."))
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
        M::AbstractManifold,
        a,
        Xc,
        A::AbstractAtlas,
        i0;
        solver=AutoVern9(Rodas5()),
        final_time=1.0,
        check_chart_switch_kwargs=NamedTuple(),
        kwargs...,
    )

Solve geodesic ODE on a manifold `M` from point of coordinates `a` in chart `i0` from an
[`AbstractAtlas`](@ref) `A` in direction of coordinates `Xc` in the induced basis.
"""
function solve_chart_exp_ode(
    M::AbstractManifold,
    a,
    Xc,
    A::AbstractAtlas,
    i0;
    solver=AutoVern9(Rodas5()),
    final_time=1.0,
    check_chart_switch_kwargs=NamedTuple(),
    kwargs...,
)
    u0 = ArrayPartition(copy(a), copy(Xc))
    cur_i = i0
    # callback stops solver when we get too close to chart boundary
    cb = FunctionCallingCallback(
        IntegratorTerminatorNearChartBoundary(check_chart_switch_kwargs);
        func_start=false,
    )
    retcode = :Terminated
    init_time = zero(final_time)
    sols = StitchedChartSolution(M, A, :Exp, typeof(i0))
    while retcode === :Terminated && init_time < final_time
        params = (M, A, cur_i)
        prob =
            ODEProblem(chart_exp_problem, u0, (init_time, final_time), params; callback=cb)
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]::typeof(final_time)
        push!(sols.sols, (sol, cur_i))
        # here we switch charts
        a_final = sol.u[end].x[1]::typeof(a)
        new_i = get_chart_index(M, A, cur_i, a_final)
        transition_map!(M, u0.x[1], A, cur_i, new_i, a_final)
        transition_map_diff!(
            M,
            u0.x[2],
            A,
            cur_i,
            a_final,
            sol.u[end].x[2]::typeof(Xc),
            new_i,
        )
        cur_i = new_i
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
        M::AbstractManifold,
        a,
        Xc,
        A::AbstractAtlas,
        i0,
        Yc;
        solver=AutoVern9(Rodas5()),
        check_chart_switch_kwargs=NamedTuple(),
        final_time=1.0,
        kwargs...,
    )

Parallel transport vector with coordinates `Yc` along geodesic on a manifold `M` from point of
coordinates `a` in a chart `i0` from an [`AbstractAtlas`](@ref) `A` in direction of
coordinates `Xc` in the induced basis.
"""
function solve_chart_parallel_transport_ode(
    M::AbstractManifold,
    a,
    Xc,
    A::AbstractAtlas,
    i0,
    Yc;
    solver=AutoVern9(Rodas5()),
    final_time=1.0,
    check_chart_switch_kwargs=NamedTuple(),
    kwargs...,
)
    u0 = ArrayPartition(copy(a), copy(Xc), copy(Yc))
    cur_i = i0
    # callback stops solver when we get too close to chart boundary
    cb = FunctionCallingCallback(
        IntegratorTerminatorNearChartBoundary(check_chart_switch_kwargs);
        func_start=false,
    )
    retcode = :Terminated
    init_time = zero(final_time)
    sols = StitchedChartSolution(M, A, :PT, typeof(i0))
    while retcode === :Terminated && init_time < final_time
        params = (M, A, cur_i)
        prob =
            ODEProblem(chart_pt_problem, u0, (init_time, final_time), params; callback=cb)
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]::typeof(final_time)
        push!(sols.sols, (sol, cur_i))
        # here we switch charts
        a_final = sol.u[end].x[1]::typeof(a)
        new_i = get_chart_index(M, A, cur_i, a_final)
        transition_map!(M, u0.x[1], A, cur_i, new_i, a_final)
        transition_map_diff!(
            M,
            u0.x[2],
            A,
            cur_i,
            a_final,
            sol.u[end].x[2]::typeof(Xc),
            new_i,
        )
        transition_map_diff!(
            M,
            u0.x[3],
            A,
            cur_i,
            a_final,
            sol.u[end].x[3]::typeof(Yc),
            new_i,
        )
        cur_i = new_i
    end
    return sols
end
