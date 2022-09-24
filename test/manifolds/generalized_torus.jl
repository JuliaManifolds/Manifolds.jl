using Revise
using Manifolds
using GLMakie, Makie
using OrdinaryDiffEq
using Test

using StaticArrays
using DiffEqCallbacks
using SciMLBase
using RecursiveArrayTools
using Manifolds: TFVector

@testset "Torus in ℝ³" begin
    M = Manifolds.TorusInR3(3, 2)
    A = Manifolds.DefaultTorusAtlas()

    #p0x = [0.5, -1.2]
    p0x = [pi / 2, 0.0]
    X_p0x = [-1.2, 0.4]
    p = [Manifolds._torus_param(M, p0x...)...]
    i_p0x = Manifolds.get_chart_index(M, A, p)
    B = induced_basis(M, A, i_p0x)
    X = get_vector(M, p, X_p0x, B)
    @test get_coordinates(M, p, X, B) ≈ X_p0x

    @test norm(X) ≈ norm(M, [0.0, 0.0], TFVector(X_p0x, B))
end

function plot_torus()
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

    fig = Figure(resolution=(1400, 1000), fontsize=22)
    ax = LScene(fig[1, 1], show_axis=true)
    gcs = [gaussian_curvature(M, p) for p in param_points]

    gcs_mm = max(abs(minimum(gcs)), abs(maximum(gcs)))

    pltobj = surface!(
        ax,
        X1,
        Y1,
        Z1;
        shading=true,
        ambient=Vec3f(0.65, 0.65, 0.65),
        backlight=1.0f0,
        color=gcs,
        colormap=Reverse(:RdBu),
        colorrange=(-gcs_mm, gcs_mm),
        transparency=true,
    )
    wireframe!(ax, X1, Y1, Z1; transparency=true, color=:gray, linewidth=0.5)
    zoom!(ax.scene, cameracontrols(ax.scene), 0.98)

    Colorbar(fig[1, 2], pltobj, height=Relative(0.5), label="Gaussian curvature")

    t_end = 200.0

    sg = SliderGrid(
        fig[2, 1],
        (label="θₚ", range=(-pi):(pi / 200):pi, startvalue=pi / 10),
        (label="φₚ", range=(-pi):(pi / 200):pi, startvalue=0.0),
        (label="θₓ", range=(-pi):(pi / 200):pi, startvalue=pi / 10),
        (label="φₓ", range=(-pi):(pi / 200):pi, startvalue=pi / 10),
        (label="θy", range=(-pi):(pi / 200):pi, startvalue=pi / 10),
        (label="φy", range=(-pi):(pi / 200):pi, startvalue=pi / 10),
    )
    A = Manifolds.DefaultTorusAtlas()

    # a point and tangent vector

    function solve_for(p0x, X_p0x, Y_transp)
        p = [Manifolds._torus_param(M, p0x...)...]
        i_p0x = Manifolds.get_chart_index(M, A, p)
        B = induced_basis(M, A, i_p0x)
        X = get_vector(M, p, X_p0x, B)
        p_exp = Manifolds.solve_chart_parallel_transport_ode(
            M,
            [0.0, 0.0],
            X_p0x,
            A,
            i_p0x,
            Y_transp;
            final_time=t_end,
        )
        return p_exp
    end

    dt = 0.1

    geo = lift(
        sg.sliders[1].value,
        sg.sliders[2].value,
        sg.sliders[3].value,
        sg.sliders[4].value,
        sg.sliders[5].value,
        sg.sliders[6].value,
    ) do θₚ, φₚ, θₓ, φₓ, θy, φy
        p_exp = solve_for([θₚ, φₚ], [θₓ, φₓ], [θy, φy])

        samples = p_exp(0.0:dt:t_end)
        return samples
    end

    geo_ps = lift(geo) do samples
        return [Point3f(s[1]) for s in samples]
    end

    # geo_Xs = lift(geo) do samples
    #     return [Point3f(s[2]) for s in samples]
    # end

    pt_indices = 1:20:length(geo[])
    geo_ps_pt = lift(geo) do samples
        return [Point3f(s[1]) for s in samples[pt_indices]]
    end
    geo_Ys = lift(geo) do samples
        return [Point3f(s[3]) for s in samples[pt_indices]]
    end

    lines!(geo_ps; linewidth=2.0, color=:red)
    arrows!(ax, geo_ps_pt, geo_Ys, linecolor=:green, arrowcolor=:green, linewidth=0.05)

    return fig
end
