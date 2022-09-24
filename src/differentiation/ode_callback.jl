
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
struct StitchedChartSolution{TM<:AbstractManifold,TA<:AbstractAtlas,TChart}
    M::TM
    A::TA
    sols::Vector{Tuple{OrdinaryDiffEq.SciMLBase.AbstractODESolution,TChart}}
end

function StitchedChartSolution(M::AbstractManifold, A::AbstractAtlas, TChart)
    return StitchedChartSolution{typeof(M),typeof(A),TChart}(M, A, [])
end

function (scs::StitchedChartSolution)(t::Real)
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

function (scs::StitchedChartSolution)(t::AbstractArray)
    return map(scs, t)
end

function solve_chart_exp_ode(
    M::AbstractManifold,
    p,
    X,
    A::AbstractAtlas,
    i0;
    solver=AutoVern9(Rodas5()),
    final_time=1.0,
    kwargs...,
)
    u0 = ArrayPartition(p, X)
    cur_i = i0
    # callback stops solver when we get too close to chart boundary
    cb = FunctionCallingCallback(maybe_switch_chart; func_start=false)
    retcode = :Terminated
    init_time = 0.0
    sols = StitchedChartSolution(M, A, typeof(i0))
    while retcode === :Terminated && init_time < final_time
        B = induced_basis(M, A, cur_i)
        params = (M, B)
        prob =
            ODEProblem(chart_exp_problem, u0, (init_time, final_time), params; callback=cb)
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]
        push!(sols.sols, (sol, cur_i))
        # here we switch charts
        new_i = get_chart_index(M, A, cur_i, sol.u[end].x[1])
        new_p0 = transition_map(M, A, cur_i, new_i, sol.u[end].x[1])
        new_B = induced_basis(M, A, new_i)
        p_final = get_point(M, A, cur_i, sol.u[end].x[1])
        new_X0 =
            get_coordinates(M, p_final, get_vector(M, p_final, sol.u[end].x[2], B), new_B)
        u0 = ArrayPartition(new_p0, new_X0)
        cur_i = new_i
    end
    return sols
end
