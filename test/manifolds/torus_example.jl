using Revise
using Manifolds
using RecursiveArrayTools
using GLMakie, Makie
using OrdinaryDiffEq
using DiffEqCallbacks
using BoundaryValueDiffEq

GLMakie.activate!()

function plot_torus()

    # selected torus
    M = Manifolds.EmbeddedTorus(3, 2)

    ϴs = LinRange(-π, π, 50)
    φs = LinRange(-π, π, 50)
    param_points = [Manifolds._torus_param(M, θ, φ) for θ in ϴs, φ in φs]
    X1 = [p[1] for p in param_points]
    Y1 = [p[2] for p in param_points]
    Z1 = [p[3] for p in param_points]

    fig = Figure(resolution=(1400, 1000), fontsize=18)
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
        (label="geodesic - θ₁", range=(-pi):(pi / 200):pi, startvalue=pi / 10),
        (label="geodesic - φ₁", range=(-pi):(pi / 200):pi, startvalue=0.0),
        (label="geodesic - θ₂", range=(-pi):(pi / 200):pi, startvalue=-pi / 3),
        (label="geodesic - φ₂", range=(-pi):(pi / 200):pi, startvalue=pi / 2);
        height=Auto(0.2f0),
    )
    rowgap!(sg.layout, 5)
    A = Manifolds.DefaultTorusAtlas()

    # a point and tangent vector

    function solve_for(p0x, X_p0x, Y_transp)
        p = [Manifolds._torus_param(M, p0x...)...]
        i_p0x = Manifolds.get_chart_index(M, A, p)
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

    # draw a geodesic between two points
    geo_r = lift(
        sg.sliders[7].value,
        sg.sliders[8].value,
        sg.sliders[9].value,
        sg.sliders[10].value,
    ) do θ₁, φ₁, θ₂, φ₂
        bvp_i = (0, 0)
        bvp_a1 = [θ₁, φ₁]
        bvp_a2 = [θ₂, φ₂]
        bvp_sol = Manifolds.solve_chart_log_bvp(M, bvp_a1, bvp_a2, A, bvp_i)
        bvp_sol_pts =
            [Point3f(get_point(M, A, bvp_i, p[1:2])) for p in bvp_sol(0.0:0.05:1.0)]
        return bvp_sol_pts
    end

    lines!(geo_r; linewidth=2.0, color=:orange)

    return fig
end
