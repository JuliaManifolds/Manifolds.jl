using Revise
using Manifolds
using GLMakie
using OrdinaryDiffEq
using Test

using StaticArrays
using DiffEqCallbacks
using SciMLBase
using RecursiveArrayTools

@testset "Torus in ℝ³" begin
    M = Manifolds.TorusInR3(3, 2)
    A = Manifolds.DefaultTorusAtlas()

    p0x = [0.5, -1.2]
    X_p0x = [-1.2, 0.4]
    p = [Manifolds._torus_param(M, p0x...)...]
    i_p0x = Manifolds.get_chart_index(M, A, p)
    B = induced_basis(M, A, i_p0x)
    X = get_vector(M, p, X_p0x, B)
    @test get_coordinates(M, p, X, B) ≈ X_p0x
end

function plot_thing()
    # take a look at http://www.rdrop.com/~half/math/torus/torus.geodesics.pdf

    GLMakie.activate!()

    # selected torus
    M = Manifolds.TorusInR3(3, 2)

    ϴs = LinRange(-π, π, 50)
    φs = LinRange(-π, π, 50)
    param_points = [Manifolds._torus_param(M, θ, φ) for θ in ϴs, φ in φs]
    X1 = [p[1] for p in param_points]
    Y1 = [p[2] for p in param_points]
    Z1 = [p[3] for p in param_points]

    fig = Figure(resolution=(1200, 800), fontsize=22)
    ax = LScene(fig[1, 1], show_axis=true)
    pltobj = surface!(
        ax,
        X1,
        Y1,
        Z1;
        shading=true,
        ambient=Vec3f(0.65, 0.65, 0.65),
        backlight=1.0f0,
        color=sqrt.(X1 .^ 2 .+ Y1 .^ 2 .+ Z1 .^ 2),
        colormap=Reverse(:bone_1),
        transparency=true,
    )
    wireframe!(ax, X1, Y1, Z1; transparency=true, color=:gray, linewidth=0.5)
    zoom!(ax.scene, cameracontrols(ax.scene), 0.98)

    # a point and tangent vector

    A = Manifolds.DefaultTorusAtlas()

    p0x = [0.5, -1.2]
    X_p0x = [-1.2, 0.4]
    p = [Manifolds._torus_param(M, p0x...)...]
    i_p0x = Manifolds.get_chart_index(M, A, p)
    B = induced_basis(M, A, i_p0x)
    X = get_vector(M, p, X_p0x, B)

    t_end = 100.0
    p_exp = Manifolds.solve_chart_exp_ode(M, [0.0, 0.0], X_p0x, A, i_p0x, final_time=t_end)
    samples = p_exp(0.0:0.1:t_end)
    geo_ps = [Point3f(s[1]) for s in samples]
    geo_Xs = [Point3f(s[2]) for s in samples]

    arrows!(ax, geo_ps, geo_Xs, linecolor=:red, arrowcolor=:red, linewidth=0.05)
    return fig
end

function maybe_switch_chart(u, t, integrator)
    (M, B) = integrator.p
    dist = norm(u.x[1] - SVector{2}(B.i))
    if dist > 2 / 3 * Manifolds.inverse_chart_injectivity_radius(M, B.A, B.i)
        # switch charts
        terminate!(integrator)
    end
    return u
end

struct StitchedChartSolution{TM<:AbstractManifold,TA<:AbstractAtlas,TChart}
    M::TM
    A::TA
    sols::Vector{Tuple{SciMLBase.AbstractODESolution,TChart}}
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

function Manifolds.solve_chart_exp_ode(
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
    cb = FunctionCallingCallback(maybe_switch_chart; func_start=false)
    retcode = :Terminated
    init_time = 0.0
    sols = StitchedChartSolution(M, A, typeof(i0))
    while retcode === :Terminated && init_time < final_time
        B = induced_basis(M, A, cur_i)
        params = (M, B)
        println((init_time, final_time))
        prob = ODEProblem(
            Manifolds.chart_exp_problem,
            u0,
            (init_time, final_time),
            params;
            callback=cb,
        )
        sol = solve(prob, solver; kwargs...)
        retcode = sol.retcode
        init_time = sol.t[end]
        push!(sols.sols, (sol, cur_i))
        new_i = Manifolds.get_chart_index(M, A, cur_i, sol.u[end].x[1])
        new_p0 = Manifolds.transition_map(M, A, cur_i, new_i, sol.u[end].x[1])
        new_B = induced_basis(M, A, new_i)
        p_final = get_point(M, A, cur_i, sol.u[end].x[1])
        new_X0 =
            get_coordinates(M, p_final, get_vector(M, p_final, sol.u[end].x[2], B), new_B)
        u0 = ArrayPartition(new_p0, new_X0)
        cur_i = new_i
    end
    return sols
end
