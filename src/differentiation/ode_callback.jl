
"""
    maybe_switch_chart(u, t, integrator)

Terminate integration when integrator goes too closely to chart boundary.
"""
function maybe_switch_chart(u, t, integrator)
    (M, B) = integrator.p
    if check_chart_switch(M, B.A, B.i, u.x[1])
        # switch charts
        OrdinaryDiffEq.terminate!(integrator)
    end
    return u
end

"""
    StitchedChartSolution{TM<:AbstractManifold,TA<:AbstractAtlas,TChart}

Solution of an ODE on manifold `M` in charts of atlas `A`.
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
    M, B = params
    p = u.x[1]
    dx = u.x[2]
    ddx = -affine_connection(M, p, dx, dx, B)
    return ArrayPartition(dx, ddx)
end

"""
    solve_chart_exp_ode(
        M::AbstractManifold,
        a,
        X,
        A::AbstractAtlas,
        i0;
        solver=AutoVern9(Rodas5()),
        final_time=1.0,
        kwargs...,
    )

Solve geodesic ODE on manifold `M` from point of coordinates `a` in chart `i0` from atlas
`A` in direction of coordinates `X` in induced basis.
"""
function solve_chart_exp_ode(
    M::AbstractManifold,
    a,
    X,
    A::AbstractAtlas,
    i0;
    solver=AutoVern9(Rodas5()),
    final_time=1.0,
    kwargs...,
)
    u0 = ArrayPartition(a, X)
    cur_i = i0
    # callback stops solver when we get too close to chart boundary
    cb = FunctionCallingCallback(maybe_switch_chart; func_start=false)
    retcode = :Terminated
    init_time = zero(final_time)
    sols = StitchedChartSolution(M, A, :Exp, typeof(i0))
    while retcode === :Terminated && init_time < final_time
        B = induced_basis(M, A, cur_i)
        params = (M, B)
        prob =
            ODEProblem(chart_exp_problem, u0, (init_time, final_time), params; callback=cb)
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]::typeof(final_time)
        push!(sols.sols, (sol, cur_i))
        # here we switch charts
        a_final = sol.u[end].x[1]::typeof(a)
        new_i = get_chart_index(M, A, cur_i, a_final)
        new_p0 = transition_map(M, A, cur_i, new_i, a_final)
        new_B = induced_basis(M, A, new_i)
        p_final = get_point(M, A, cur_i, a_final)
        new_X0 = get_coordinates(
            M,
            p_final,
            get_vector(M, p_final, sol.u[end].x[2]::typeof(X), B),
            new_B,
        )
        u0 = ArrayPartition(new_p0, new_X0)::typeof(u0)
        cur_i = new_i
    end
    return sols
end

function chart_pt_problem(u, params, t)
    M, B = params
    p = u.x[1]
    dx = u.x[2]
    dY = u.x[3]
    ddx = -affine_connection(M, p, dx, dx, B)
    ddY = -affine_connection(M, p, dx, dY, B)
    return ArrayPartition(dx, ddx, ddY)
end

"""

Parallel transport vector `Y` along geodesic on manifold `M` from point of coordinates `a`
in chart `i0` from atlas `A` in direction of coordinates `X` in induced basis.
"""
function solve_chart_parallel_transport_ode(
    M::AbstractManifold,
    a,
    X,
    A::AbstractAtlas,
    i0,
    Y;
    solver=AutoVern9(Rodas5()),
    final_time=1.0,
    kwargs...,
)
    u0 = ArrayPartition(a, X, Y)
    cur_i = i0
    # callback stops solver when we get too close to chart boundary
    cb = FunctionCallingCallback(maybe_switch_chart; func_start=false)
    retcode = :Terminated
    init_time = zero(final_time)
    sols = StitchedChartSolution(M, A, :PT, typeof(i0))
    while retcode === :Terminated && init_time < final_time
        B = induced_basis(M, A, cur_i)
        params = (M, B)
        prob =
            ODEProblem(chart_pt_problem, u0, (init_time, final_time), params; callback=cb)
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]::typeof(final_time)
        push!(sols.sols, (sol, cur_i))
        # here we switch charts
        a_final = sol.u[end].x[1]::typeof(a)
        new_i = get_chart_index(M, A, cur_i, a_final)
        new_p0 = transition_map(M, A, cur_i, new_i, a_final)
        new_B = induced_basis(M, A, new_i)
        p_final = get_point(M, A, cur_i, a_final)
        new_X0 = get_coordinates(
            M,
            p_final,
            get_vector(M, p_final, sol.u[end].x[2]::typeof(X), B),
            new_B,
        )
        new_Y0 = get_coordinates(
            M,
            p_final,
            get_vector(M, p_final, sol.u[end].x[3]::typeof(Y), B),
            new_B,
        )
        u0 = ArrayPartition(new_p0, new_X0, new_Y0)::typeof(u0)
        cur_i = new_i
    end
    return sols
end
