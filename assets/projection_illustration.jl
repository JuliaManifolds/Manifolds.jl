using Manifolds, LinearAlgebra, PGFPlotsX, Colors

out_file = joinpath(
    @__DIR__,
    "..",
    "docs",
    "src",
    "assets",
    "images",
    "projection_illustration.png",
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

p = (-2 + 1im) / abs(-2 + 1im)
X = (imag(p) - real(p) * im)
q = -1.3 + 1.5im

s1 = project(M, q)
s2 = project(M, p, q)

circle = [[sin(φ), cos(φ)] for φ in range(0, 2π, length=n)]
tp = @pgf Axis({
    axis_lines = "none",
    axis_equal,
    xmin = -1.7,
    xmax = 1.7,
    ymin = -.25,
    ymax = 1.4,
    scale = 1.6,
})

plain_opts = @pgf {"no markers", color = light_color, line_width = base_linewidth}

push!(tp, Plot(plain_opts, Coordinates(Tuple.(circle))))
plaint_opts_a = @pgf {"only marks", mark_size = geo_marker_size / 2, color = light_color}
push!(tp, Plot(plaint_opts_a, Coordinates(Tuple.([[0.0, 0.0]]))))

tangent_line2 = [C_to_2D(p - 0.8 * X), C_to_2D(p + 1.1 * X)]
push!(tp, Plot(plain_opts, Coordinates(Tuple.(tangent_line2))))

projection_opts2 = @pgf {
    "no markers",
    "densely dotted",
    mark_size = projection_marker_size,
    color = projection_color2,
}
projection_line2 = [C_to_2D(q), C_to_2D(s1)]
push!(tp, Plot(projection_opts2, Coordinates(Tuple.(projection_line2))))
projection_opts2_a =
    @pgf {"only marks", mark_size = projection_marker_size, color = projection_color2}
push!(tp, Plot(projection_opts2_a, Coordinates(Tuple.([C_to_2D(s1)]))))

geo_opts2 = @pgf {"only marks", mark_size = geo_marker_size, color = geo_color}
push!(tp, Plot(geo_opts2, Coordinates(Tuple.(C_to_2D.([p, q])))))

projection_opts3 = @pgf {
    "no markers",
    "densely dotted",
    mark_size = projection_marker_size,
    color = projection_color3,
}
projection_line3 = [C_to_2D(q), C_to_2D(p + s2)]
push!(tp, Plot(projection_opts3, Coordinates(Tuple.(projection_line3))))
projection_opts3_a =
    @pgf {"only marks", mark_size = projection_marker_size, color = projection_color3}
push!(tp, Plot(projection_opts3_a, Coordinates(Tuple.([C_to_2D(p + s2)]))))

push!(
    tp,
    raw"\node[label ={[label distance=.05cm]below:{\color{gray}$q$}}] at (axis cs:" *
    "$(real(q)),$(imag(q))) {};",
)
push!(
    tp,
    raw"\node[label ={[label distance=.05cm]left:{\color{gray}$p$}}] at (axis cs:" *
    "$(real(p)),$(imag(p))) {};",
)
# push!(tp, raw"\node[label ={[label distance=.05cm]right:{$s_1=\mathrm{proj}_{\mathcal C}(q_2)=$\texttt{project(M,q2)}}}] at (axis cs:"*"$(real(s1)),$(imag(s1))) {};")
push!(
    tp,
    raw"\node[label ={[label distance=.05cm]right:{\color{gray}$s_1=$\texttt{project(M,q)$\in\mathcal C$}}}] at (axis cs:" *
    "$(real(s1)),$(imag(s1))) {};",
)
push!(
    tp,
    raw"\node[label ={[label distance=.05cm]0:{\color{gray}$s_2=$\texttt{project(M,p,q)}$\in T_{p}\mathcal C$}}] at (axis cs:" *
    "$(real(p+s2)),$(imag(p+s2))) {};",
)
push!(tp, raw"\node at (axis cs: 0,-.6) {\color{gray}$\mathcal C$};")

pgfsave(out_file, tp, dpi=1200)
