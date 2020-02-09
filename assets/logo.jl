using Manifolds, LinearAlgebra, PGFPlotsX, Colors, Distributions, Contour, Random

#
# Settings
#
plain = false # in plain mode small details are left out
dark_mode = false && !plain # the plain version is always bright

line_offset_brightness = 0.25
patch_opacity = 0.8
tangent_space_opacity = 0.8
line_width = 6
geo_line_width = 12 * ( (plain) ? 3 : 1)
mesh_line_width = line_width / 3
basis_color = dark_mode ? RGB(0.33, 0.33, 0.33) : RGB(0.67, 0.67, 0.67)
inner_prod_color = dark_mode ? RGB(0.67, 0.67, 0.67) : RGB(0.33, 0.33, 0.33)
logo_colors = [(77, 100, 174), (57, 151, 79), (202, 60, 50), (146, 89, 163)] # Julia colors

vector_color = dark_mode ? "white" : "black"
point_color = vector_color
rgb_logo_colors = map(x -> RGB(x ./ 255...), logo_colors)
rgb_logo_colors_bright = map(x -> RGB( (1 + line_offset_brightness) .* x ./ 255...), logo_colors)
rgb_logo_colors_dark = map(x -> RGB( (1 - line_offset_brightness) .* x ./ 255...), logo_colors)

out_file_prefix = plain ? "logo_plain" : (dark_mode ? "logo-dark" : "logo")
out_file_ext = ".pdf"

#
# Helping functions
#
polar_to_cart(r, θ) = (r * cos(θ), r * sin(θ))

cart_to_polar(x, y) = (hypot(x, y), atan(y, x))

function logdetjacexp(S::Sphere, x, v, B)
    vⁱ = get_coordinates(S, x, v, B)
    r, θ = cart_to_polar(vⁱ...)
    return log(Manifolds.usinc(r))
end

function normal_coord_to_vector(M, x, rθ, B)
    nc = collect(polar_to_cart(rθ...))
    v = get_vector(M, x, nc, B)
    return v
end

function normal_coord_to_vector_at_point(M, x, rθ, B)
    v = normal_coord_to_vector(M, x, rθ, B)
    return v + x
end

function normal_coord_to_point(M, x, rθ, B)
    v = normal_coord_to_vector(M, x, rθ, B)
    return exp(M, x, v)
end

function plot_normal_coord!(
    ax,
    M,
    x,
    B,
    rs,
    θs;
    ncirc = 9,
    tangent = false,
    options = Dict(),
    kwargs...,
)
    f = tangent ? normal_coord_to_vector_at_point : normal_coord_to_point
    for r in rs[2:end-1]
        push!(ax, Plot3(options, Coordinates(map(θ -> Tuple(f(M, x, [r, θ], B)), θs))))
    end
    for θ in range(0, 2π; length = ncirc)
        push!(ax, Plot3(options, Coordinates(map(r -> Tuple(f(M, x, [r, θ], B)), rs))))
    end
    return ax
end

function plot_patch!(ax, M, x, B, r, θs; tangent = false, options = Dict())
    f = tangent ? normal_coord_to_vector_at_point : normal_coord_to_point
    push!(ax, Plot3(options, Coordinates(map(θ -> Tuple(f(M, x, [r, θ], B)), θs))))
    return ax
end

function plot_geodesic!(ax, M, x, y; n = 100, options = Dict())
    γ = shortest_geodesic(M, x, y)
    T = range(0, 1; length = n)
    push!(ax, Plot3(options, Coordinates(Tuple.(γ(T)))))
    return ax
end

function plot_tangent_vector!(ax, M, x, v; n = 100, options = nothing, color = "black")
    if options === nothing
        options = @pgf {
            quiver = {
                u = "\\thisrow{u}",
                v = "\\thisrow{v}",
                w = "\\thisrow{w}",
                "every arrow/.append style={-{Latex[scale length=\\pgfplotspointmetatransformed/1000]}}",
            },
            "-stealth",
            roundcaps,
            color = color,
            line_width = 4,
        }
    end
    @pgf push!(
        ax,
        Plot3(
            options,
            Table(x = [x[1]], y = [x[2]], z = [x[3]], u = [v[1]], v = [v[2]], w = [v[3]]),
        ),
    )
    return ax
end

function plot_inner_product!(ax, M, x, v, B; options = Dict())
    es = get_vectors(M, x, B)
    @assert length(es) == 2
    coords = get_coordinates(M, x, v, B)
    vs = coords .* es .+ Ref(x)
    for vi in vs
        push!(ax, Plot3(options, Coordinates([Tuple(vi), Tuple(v + x)])))
    end
    return ax
end

#
# Prepare document
#
resize!(PGFPlotsX.CUSTOM_PREAMBLE, 0)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\pgfplotsset{scale=6.0}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usetikzlibrary{arrows.meta}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\pgfplotsset{roundcaps/.style={line cap=round}}")
push!(
    PGFPlotsX.CUSTOM_PREAMBLE,
    raw"\pgfplotsset{circledotted/.style={dash pattern=on 0pt off 3\pgflinewidth, line cap=round}}",
)
if dark_mode
    push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\pagecolor{black}")
end
S = Sphere(2)

center = normalize([1, 1, 1])
x, y, z = eachrow(Matrix{Float64}(I, 3, 3))
γ1 = shortest_geodesic(S, center, z)
γ2 = shortest_geodesic(S, center, x)
γ3 = shortest_geodesic(S, center, y)
p1 = γ1(1)
p2 = γ2(1)
p3 = γ3(1)

#
# Setup Axes
if dark_mode
    tp = @pgf Axis({
        axis_lines = "none",
        axis_equal,
        view = "{135}{35}",
        #"axis_background/.style" = {fill = "black"},
        zmin = -0.05, zmax = 1.0, xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0
    })
else
    tp = @pgf Axis({
        axis_lines = "none",
        axis_equal,
        view = "{135}{35}",
        zmin = -0.05, zmax = 1.0, xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0
    })
end
rs = range(0, π / 5; length = 6)
θs = range(0, 2π; length = 100)

#
# Plot manifold patches
patch_colors = rgb_logo_colors[2:end]
patch_colors_line = dark_mode ? rgb_logo_colors_bright[2:end] : rgb_logo_colors_dark[2:end]
patch_colors_contour = dark_mode ? rgb_logo_colors_dark[2:end] : rgb_logo_colors_bright[2:end]
base_points = [p1, p2, p3]
basis_vectors = [log(S, p1, p2), log(S, p2, p1), log(S, p3, p1)]
for i in eachindex(base_points)
    x = base_points[i]
    B = DiagonalizingOrthonormalBasis(basis_vectors[i])
    basis = get_basis(S, x, B)
    optionsP = @pgf {fill = patch_colors[i], draw = "none", opacity = patch_opacity}
    plot_patch!(tp, S, x, basis, π / 5, θs; options = optionsP)
    optionsL = @pgf {
        circledotted,
        color = patch_colors_line[i],
        line_width = mesh_line_width,
        opacity = 1.,
    }
    plot_normal_coord!(tp, S, x, basis, rs, θs; options = optionsL)
end

#
# Plot geodesics
options = @pgf {no_markers, roundcaps, line_width = geo_line_width, color = rgb_logo_colors[1]}
plot_geodesic!(tp, S, base_points[1], base_points[2]; options = options)
plot_geodesic!(tp, S, base_points[1], base_points[3]; options = options)
plot_geodesic!(tp, S, base_points[2], base_points[3]; options = options)

if !plain
    #
    # Plot parallel transported vectors
    x = base_points[2]
    e1 = normalize(log(S, x, base_points[1]))
    e2 = normalize(log(S, x, base_points[3]))
    B = DiagonalizingOrthonormalBasis(e1)
    basis = get_basis(S, x, B)
    v = get_vector(S, x, π / 5 .* [0.4, 0.7], basis)
    if (!plain)
        d = log(S, x, base_points[1])
        for t in range(1 ./5 , 1; length = 5)
            te = t .* d
            tx = exp(S, x, te)
            tv = vector_transport_direction(S, x, v, te)
            plot_tangent_vector!(tp, S, tx, tv; color = vector_color)
        end
    end

    #
    # Plot tangent plane and its conent
    B = DiagonalizingOrthonormalBasis(basis_vectors[2])
    basis = get_basis(S, x, B)
    optionsP = @pgf {
        fill = patch_colors[2],
        draw = "none",
        opacity = tangent_space_opacity,
    }
    plot_patch!(tp, S, x, basis, π / 5, θs; options = optionsP, tangent = true)
    optionsL = @pgf {
        color = rgb_logo_colors_dark[3],
        line_width = mesh_line_width,
        opacity = 0.5*tangent_space_opacity,

    }
    plot_normal_coord!(tp, S, x, basis, rs, θs; options = optionsL, tangent = true)

    #
    # plot tangent space content on left plane
    x = base_points[2]
    e1 = normalize(log(S, x, base_points[1]))
    e2 = normalize(log(S, x, base_points[3]))
    options = @pgf {
        circledotted,
        line_width = line_width,
        opacity = tangent_space_opacity,
        color = inner_prod_color,
    }
    plot_inner_product!(tp, S, x, v, basis; options = options)
    plot_tangent_vector!(tp, S, x, e1 .* π / 5; color = basis_color)
    plot_tangent_vector!(tp, S, x, e2 .* π / 5; color = basis_color)
    plot_tangent_vector!(tp, S, x, v; color = vector_color)

    #
    # Plot distribution in right patch
    Random.seed!(10)
    x = base_points[3]
    B = DiagonalizingOrthonormalBasis(basis_vectors[3])
    basis = get_basis(S, x, B)
    vxs = range(-1, 1; length = 100) .* π / 5
    vys = range(-1, 1; length = 100) .* π / 5
    d = MvNormal([0.02, 0.06], diagm([0.05, 0.1] .* π / 5))
    logp =
        (
            (vx, vy) ->
                logpdf(d, [vx, vy]) -
                logdetjacexp(S, x, get_vector(S, x, [vx, vy], basis), basis)
        ).(vxs, vys')
    cs = Contour.contours(vxs, vys, exp.(logp), 6)
    for level in cs.contours
        for line in Contour.lines(level)
            coords = map(line.vertices) do vⁱ
                v = get_vector(S, x, vⁱ, basis)
                y = exp(S, x, v)
                return Tuple(y)
            end
            @pgf push!(
                tp,
                Plot3(
                    {
                        fill = patch_colors_contour[3],
                        color = patch_colors_line[3],
                        fill_opacity = 0.3,
                        draw_opacity = 0.6,
                        line_width = mesh_line_width,
                    },
                    Coordinates(coords),
                ),
            )
        end
    end
    rand_pts = map(vⁱ -> exp(S, x, get_vector(S, x, vⁱ, basis)), eachcol(rand(d, 100)))
    @pgf push!(
        tp,
        Plot3(
            {
                only_marks,
                mark_options = {fill = point_color, draw_opacity = 0},
                opacity = 0.66,
                draw = "none",
            },
            Coordinates(Tuple.(rand_pts)),
        ),
    )
end
#
# Export Logo.
out_file = "$(out_file_prefix)$(out_file_ext)"
pgfsave(out_file, tp)
