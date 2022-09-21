
using Manifolds
using GLMakie
using OrdinaryDiffEq

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
    X_p0x = [-0.2, 0.4]
    p = [Manifolds._torus_param(M, p0x...)...]
    i_p0x = Manifolds.get_chart_index(M, A, p)
    B = induced_basis(M, A, i_p0x, Manifolds.TangentSpaceType())
    X = get_vector(M, p, X_p0x, B)

    arrows!(ax, [Point3f(p)], [Point3f(X)], linecolor=:red, arrowcolor=:red)

    p_exp = Manifolds.solve_chart_exp_ode(M, p0x, X_p0x, A, i_p0x)
    return fig
end
