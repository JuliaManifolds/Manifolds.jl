using Manifolds, LinearAlgebra, PGFPlotsX, Colors

out_file = joinpath(
    @__DIR__,
    "..",
    "docs",
    "src",
    "assets",
    "images",
    "retraction_illustration.png",
)
n = 100

line_width = 1 / 2
light_color = colorant"#BBBBBB"
base_linewidth = line_width
point_color = colorant"#0077BB" #Blue
main_linewidth = 1.25
tangent_color = colorant"#33BBEE" #Cyan
tangent_linewidth = main_linewidth
geo_color = colorant"#009988" #teal
geo_linewidth = 1.25
geo_marker_size = 1.5
projection_color = colorant"#EE7733"
projection_color2 = colorant"#0077BB" #Blue
projection_color3 = colorant"#33BBEE" #Cyan
projection_marker_size = 1.5

C_to_2D(p) = [real(p), imag(p)]

M = Circle(ℂ)

p = (1 + 2im) / abs(1 + 2im)
X = imag(p) - real(p) * im
qE = exp(M, p, X)
qP = project(M, p + X)

circle = [[sin(φ), cos(φ)] for φ in range(0, 2π, length=n)]
tp = @pgf Axis({
    axis_lines = "none",
    axis_equal,
    xmin = -1.7,
    xmax = 1.7,
    ymin = -0.25,
    ymax = 1.4,
    scale = 1.6,
})

plain_opts = @pgf {"no markers", color = light_color, line_width = base_linewidth}

push!(tp, Plot(plain_opts, Coordinates(Tuple.(circle))))
plaint_opts_a = @pgf {"only marks", mark_size = geo_marker_size / 2, color = light_color}
push!(tp, Plot(plaint_opts_a, Coordinates(Tuple.([[0.0, 0.0]]))))

tangent_line = [C_to_2D(p - 0.8 * X), C_to_2D(p + 1.3 * X)]
push!(tp, Plot(plain_opts, Coordinates(Tuple.(tangent_line))))

tangent_opts =
    @pgf {"no markers", "->", color = tangent_color, line_width = tangent_linewidth}
tangent_vec = [C_to_2D(p), C_to_2D(p + X)]
push!(tp, Plot(tangent_opts, Coordinates(Tuple.(tangent_vec))))

geo_opts = @pgf {"no markers", "-", color = geo_color, line_width = geo_linewidth}
geo_pts = C_to_2D.(shortest_geodesic(M, p, qE, range(0, 1, length=n)))
push!(tp, Plot(geo_opts, Coordinates(Tuple.(geo_pts))))
geo_opts2 = @pgf {"only marks", mark_size = geo_marker_size, color = geo_color}
push!(tp, Plot(geo_opts2, Coordinates(Tuple.([C_to_2D(p), C_to_2D(qE)]))))

projection_opts = @pgf {
    "no markers",
    "densely dotted",
    color = projection_color,
    line_width = base_linewidth,
}
projection_line = [C_to_2D(p + X), C_to_2D(qP)]
push!(tp, Plot(projection_opts, Coordinates(Tuple.(projection_line))))
projection_opts_a =
    @pgf {"only marks", mark_size = projection_marker_size, color = projection_color}
push!(tp, Plot(projection_opts_a, Coordinates(Tuple.([C_to_2D(qP)]))))

push!(
    tp,
    raw"\node[label ={[label distance=.05cm]above:{\color{gray}$p$}}] at (axis cs:" *
    "$(real(p)),$(imag(p))) {};",
)
push!(
    tp,
    raw"\node[label ={[label distance=.1cm]above:{\color{gray}$X\in T_p\mathcal C$}}] at (axis cs:" *
    "$(real(p+X/2)),$(imag(p+X/2))) {};",
)
push!(
    tp,
    raw"\node[label ={[label distance=.05cm]left:{\color{gray}$q'=\mathrm{retr}^{\mathrm{proj}}_pX=\mathrm{proj}_{\mathcal C}(p+X)$}}] at (axis cs:" *
    "$(real(qP)),$(imag(qP))) {};",
)
push!(
    tp,
    raw"\node[label ={[label distance=.05cm]left:{\color{gray}$q=\exp_pX$}}] at (axis cs:" *
    "$(real(qE)),$(imag(qE))) {};",
)

push!(tp, raw"\node at (axis cs: 0,-.6) {\color{gray}$\mathcal C$};")

pgfsave(out_file, tp; dpi=1200)
