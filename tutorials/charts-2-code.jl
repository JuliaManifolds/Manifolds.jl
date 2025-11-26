using Manifolds, RecursiveArrayTools, OrdinaryDiffEq, DiffEqCallbacks, BoundaryValueDiffEq

using DifferentiationInterface, ForwardDiff

using Manifolds: TangentSpaceType

using LinearAlgebra

using CairoMakie

# space outside of a black hole with Schwarzschild radius rₛ
struct BlackHoleOutside <: AbstractManifold{ℝ}
    rₛ::Float64
end

struct SchwarzschildAtlas <: AbstractAtlas{ℝ} end

Manifolds.manifold_dimension(::BlackHoleOutside) = 4
Manifolds.representation_size(::BlackHoleOutside) = (4,)

function Manifolds.affine_connection!(M::BlackHoleOutside, Zc, A::SchwarzschildAtlas, i, a, Xc, Yc)
    return Manifolds.levi_civita_affine_connection!(M, Zc, A, i, a, Xc, Yc)
end

function Manifolds.check_chart_switch(::BlackHoleOutside, A::SchwarzschildAtlas, i, a)
    return false
end

function Manifolds.inner(M::BlackHoleOutside, ::SchwarzschildAtlas, i, a, Xc, Yc)
    t, r, θ, ϕ = a
    r_block = (1 - M.rₛ / r)
    # assuming c = 1
    #@show r_block, Xc, Yc, a
    return Xc[1] * r_block * Yc[1] - Xc[2] * Yc[2] / r_block - r^2 * (Xc[3] * Yc[3] + (sin(θ)^2) * Xc[4] * Yc[4])
end


function Manifolds.get_chart_index(::BlackHoleOutside, ::SchwarzschildAtlas, p)
    return nothing
end
function Manifolds.get_chart_index(::BlackHoleOutside, ::SchwarzschildAtlas, i, a)
    return nothing
end

function Manifolds.get_parameters!(M::BlackHoleOutside, x, ::SchwarzschildAtlas, i, p)
    x[1] = p[1] # t
    r = norm(p[2:4])
    x[2] = r
    x[3] = acos(p[4] / r) # θ
    x[4] = atan(p[3], p[2]) # ϕ
    return x
end

function Manifolds.get_point!(M::BlackHoleOutside, p, ::SchwarzschildAtlas, i, x)
    p[1] = x[1]
    p[2] = x[2] * sin(x[3]) * cos(x[4])
    p[3] = x[2] * sin(x[3]) * sin(x[4])
    p[4] = x[2] * cos(x[3])
    return p
end


# simulation
function sim()
    M = BlackHoleOutside(1.0)
    p0 = [0.0, 10.0, 0.0, 0.0]

    A = SchwarzschildAtlas()
    i = nothing

    X0 = [1.0, 0.0, 0.18, 0.0]
    a_p0 = get_parameters(M, A, i, p0)
    B = induced_basis(M, A, a_p0)
    c_X0 = get_coordinates(M, p0, X0, B)
    final_time = 5000.0
    sol = Manifolds.solve_chart_exp_ode(M, a_p0, c_X0, A, i; final_time = final_time, solver = Tsit5())

    sampled_solution = sol(range(0.0, final_time; length = 20000))

    x_min = -20.0
    x_max = 20.0
    y_min = -20.0
    y_max = 20.0

    ks_samples = 300
    ks_x = range(x_min, x_max; length = ks_samples)
    ks_y = range(y_min, y_max; length = ks_samples)
    function get_ks(a_x, a_y)
        if a_x^2 + a_y^2 > M.rₛ^2
            return Manifolds.kretschmann_scalar(M, A, i, get_parameters(M, A, i, [0.0, a_x, a_y, 0.0]))
        else
            return 0.0
        end
    end

    ks_vals = [log.(get_ks(a_x, a_y)) for a_x in ks_x, a_y in ks_y]

    # plotting
    x_values = [s[1][2] for s in sampled_solution]
    y_values = [s[1][3] for s in sampled_solution]

    fig = Figure(; size = (800, 800))
    ax = Axis(fig[1, 1]; title = "2D Plot of Sampled Solution", xlabel = "x", ylabel = "y", aspect = AxisAspect(1))

    xlims!(ax, x_min, x_max)
    ylims!(ax, y_min, y_max)

    hm = heatmap!(ax, ks_x, ks_y, ks_vals; colormap = :summer) # show Kretschmann scalar

    Colorbar(fig[:, end + 1], hm)

    θ = range(0, 2π, length = 400)             # parameter for the circle
    xs = cos.(θ) .* 1.0                      # radius = 1
    ys = sin.(θ) .* 1.0

    poly!(ax, xs, ys, color = :black)        # filled polygon approximating the circle

    lines!(ax, x_values, y_values, color = :blue, label = "movement")

    axislegend(ax)
    display(fig)
    return fig
end


using Observables

function anim()
    M = BlackHoleOutside(1.0)
    p0 = [0.0, 10.0, 0.0, 0.0]

    A = SchwarzschildAtlas()
    i = nothing

    X0 = [1.0, 0.0, 0.18, 0.0]
    a_p0 = get_parameters(M, A, i, p0)
    B = induced_basis(M, A, a_p0)
    c_X0 = get_coordinates(M, p0, X0, B)
    final_time = 5000.0
    sol = Manifolds.solve_chart_exp_ode(M, a_p0, c_X0, A, i; final_time = final_time, solver = Tsit5())

    sampled_solution = sol(range(0.0, final_time; length = 5000))

    # Extract x and y values for animation
    x_values = [s[1][2] for s in sampled_solution]
    y_values = [s[1][3] for s in sampled_solution]

    # Create figure and axis
    fig = Figure(; size = (800, 800))
    ax = Axis(fig[1, 1]; title = "Animated Sampled Solution", xlabel = "x", ylabel = "y", aspect = AxisAspect(1))

    # Observables to update during the animation
    path_x = Observable(x_values[1:1])
    path_y = Observable(y_values[1:1])
    current_x = Observable([x_values[1]])
    current_y = Observable([y_values[1]])

    # Draw Schwarzschild radius
    arc!(ax, Point2f(0), M.rₛ, -π, π; color = :black)

    # Plot line (path) and moving point
    lines!(ax, path_x, path_y; color = :blue, linewidth = 2)
    scatter!(ax, current_x, current_y; color = :red, markersize = 8)
    xlims!(ax, -20, 20)
    ylims!(ax, -20, 20)

    # Record animation
    nframes = length(x_values)
    record(fig, "black_hole_orbit.mp4", 1:nframes; framerate = 30) do frame
        # update observables for this frame
        path_x[] = x_values[1:frame]
        path_y[] = y_values[1:frame]
        current_x[] = [x_values[frame]]
        current_y[] = [y_values[frame]]
    end

    return println("Animation saved as black_hole_orbit.mp4")
end

# generic stuff
