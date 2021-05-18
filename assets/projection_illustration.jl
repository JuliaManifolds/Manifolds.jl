using Manifolds, LinearAlgebra, PGFPlotsX, Colors

out_file = joinpath(@__DIR__, "projection_illustration.png")
n = 100

line_width=1
circle_color = colorant"#BBBBBB"
circle_linewidth = line_width
point_color = colorant"#0077BB" #Blue
tangent_color = colorant"#33BBEE" #Cyan
geo_color = colorant"#009988" #teal

C_to_2D(p) = [real(p),imag(p)]

M = Circle(ℂ)

p = (1+2im)/abs(1+2im)
X = imag(p)-real(p)*im

circle = [ [sin(φ), cos(φ)] for φ ∈ range(0,2π, length=n) ]
tp = @pgf Axis({
    axis_lines = "none",
    axis_equal,
    xmin = -1.2,
    xmax = 1.2,
    ymin = -1.2,
    ymax = 1.2,
})

circle_opts = @pgf { no_markers, color = circle_color, line_width=circle_linewidth }

push!(
    tp,
    Plot(circle_opts, Coordinates(Tuple.(circle)))
)

tangent_opts = @pgf { no_markers, color = tangent_color, line_width=circle_linewidth }
tangent_line = [C_to_2D(p-X),C_to_2D(p+X)]
push!(
    tp,
    Plot(circle_opts, Coordinates(Tuple.(tangent_line)))
)

pgfsave(out_file, tp)
