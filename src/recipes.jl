#
# Defaults
#
CIRCLE_DEFAULT_PLOT_POINTS = 720

#
# Plotting Recipe – Poincaré Ball
#
@recipe function f(
    M::Hyperbolic{2},
    pts::AbstractVector{P};
    circle_points = CIRCLE_DEFAULT_PLOT_POINTS,
    geodesic_interpolation = -1,
    hyperbolic_border_color = RGBA(0.0, 0.0, 0.0, 1.0),
) where {P<:PoincareBallPoint}
    @series begin
        φr = range(0, stop = 2 * π, length = circle_points)
        x = [cos(φ) for φ in φr]
        y = [sin(φ) for φ in φr]
        seriestype := :path
        seriescolor := hyperbolic_border_color
        label := ""
        x, y
    end
    show_axis --> false
    framestyle -> :none
    axis --> false
    xlims --> (-1.01, 1.01)
    ylims --> (-1.01, 1.01)
    grid --> false
    aspect_ratio --> :equal
    tickfontcolor --> RGBA(1.0, 1.0, 1.0, 1.0)
    if geodesic_interpolation < 0
        seriestype --> :scatter
        return [p.value[1] for p in pts], [p.value[2] for p in pts]
    else
        lpts = empty(pts)
        for i in 1:(length(pts) - 1)
            # push interims points on geodesics between two points.
            push!(
                lpts,
                shortest_geodesic(
                    M,
                    pts[i],
                    pts[i + 1],
                    collect(range(0, 1, length = geodesic_interpolation + 2))[1:(end - 1)], # omit end point
                )...,
            )
        end
        push!(lpts, last(pts)) # add last end point
        # split into x, y and plot as curve
        seriestype --> :path
        return [p.value[1] for p in lpts], [p.value[2] for p in lpts]
    end
end
# Tangents as quiver
@recipe function f(
    ::Hyperbolic{2},
    pts::AbstractVector{P},
    vecs::AbstractVector{T},
    circle_points = CIRCLE_DEFAULT_PLOT_POINTS,
    hyperbolic_border_color = RGBA(0.0, 0.0, 0.0, 1.0),
) where {P<:PoincareBallPoint,T<:PoincareBallTVector}
    @series begin
        φr = range(0, stop = 2 * π, length = circle_points)
        x = [cos(φ) for φ in φr]
        y = [sin(φ) for φ in φr]
        seriestype := :path
        seriescolor := hyperbolic_border_color
        label := ""
        x, y
    end
    show_axis --> false
    framestyle -> :none
    axis --> false
    xlims --> (-1.01, 1.01)
    ylims --> (-1.01, 1.01)
    grid --> false
    aspect_ratio --> :equal
    tickfontcolor --> RGBA(1.0, 1.0, 1.0, 1.0)
    seriestype := :quiver
    quiver := ([v.value[1] for v in vecs], [v.value[2] for v in vecs])
    return [p.value[1] for p in pts], [p.value[2] for p in pts]
end
#
# Plotting Recipe – Poincaré Half plane
#
@recipe function f(
    M::Hyperbolic{2},
    pts::AbstractVector{T};
    geodesic_interpolation = -1,
) where {T<:PoincareHalfSpacePoint}
    aspect_ratio --> :equal
    framestyle --> :origin
    if geodesic_interpolation < 0
        seriestype --> :scatter
        return [p.value[1] for p in pts], [p.value[2] for p in pts]
    else
        lpts = empty(pts)
        for i in 1:(length(pts) - 1)
            # push interims points on geodesics between two points.
            push!(
                lpts,
                shortest_geodesic(
                    M,
                    pts[i],
                    pts[i + 1],
                    collect(range(0, 1, length = geodesic_interpolation + 2))[1:(end - 1)], # omit end point
                )...,
            )
        end
        push!(lpts, last(pts)) # add last end point
        # split into x, y, z and plot as curve
        seriestype --> :path
        return [p.value[1] for p in lpts], [p.value[2] for p in lpts]
    end
end
# Tangents as quiver
@recipe function f(
    ::Hyperbolic{2},
    pts::AbstractVector{P},
    vecs::AbstractVector{T},
) where {P<:PoincareHalfSpacePoint,T<:PoincareHalfSpaceTVector}
    aspect_ratio --> :equal
    framestyle --> :origin
    seriestype := :quiver
    quiver := ([v.value[1] for v in vecs], [v.value[2] for v in vecs])
    return [p.value[1] for p in pts], [p.value[2] for p in pts]
end
#
# Plotting Recipe – Sphere
#
# (1) basic sphere & points
@recipe function f(
    ::Sphere{2,ℝ},
    pts::AbstractVector{P};
    wiresphere = true,
    wires = 32,
    wires_latitude = wires,
    wires_longitude = wires,
    wireframe_color = RGBA(0.0, 0.0, 0.0, 1.0),
    solidsphere = false,
    solid_resolution = 32,
    solid_resolution_latitude = solid_resolution,
    solid_resolution_longitude = solid_resolution,
    solid_color = RGBA(0.9, 0.9, 0.9, 0.8),
) where {P}
    # part I: wire
    if wiresphere
        u = range(0, 2π, length = wires_longitude + 1)
        v = range(0, π, length = wires_latitude + 1)
        x = cos.(u) * sin.(v)'
        y = sin.(u) * sin.(v)'
        z = repeat(cos.(v)',outer=[wires_longitude+1, 1])
        @series begin
            seriestype := :wireframe
            color := wireframe_color
            x, y, z
        end
    end
    # part II: solid sphere
    if solidsphere
        u = range(0, 2π, length = solid_resolution_longitude + 1)
        v = range(0, π, length = solid_resolution_latitude + 1)
        x = cos.(u) * sin.(v)'
        y = sin.(u) * sin.(v)'
        z = repeat(cos.(v)',outer=[wires_longitude + 1, 1])
        @series begin
            seriestype := :surface
            color := solid_color
            x, y, z
        end
    end
    show_axis --> false
    framestyle -> :none
    axis --> false
    xlims --> (-1.01, 1.01)
    ylims --> (-1.01, 1.01)
    zlims --> (-1.01, 1.01)
    grid --> false
    aspect_ratio --> :equal
    tickfontcolor --> RGBA(1.0, 1.0, 1.0, 1.0)
    if geodesic_interpolation < 0
        seriestype --> :scatter
        return [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]
    else
        lpts = empty(pts)
        for i in 1:(length(pts) - 1)
            # push interims points on geodesics between two points.
            push!(
                lpts,
                shortest_geodesic(
                    M,
                    pts[i],
                    pts[i + 1],
                    collect(range(0, 1, length = geodesic_interpolation + 2))[1:(end - 1)], # omit end point
                )...,
            )
        end
        push!(lpts, last(pts)) # add last end point
        # split into x, y and plot as curve
        seriestype --> :path
        return [p[1] for p in lpts], [p[2] for p in lpts], [p[3] for p in lpts]
    end
end
# (2) points on the sphere
@recipe function f(
    ::Sphere{2,ℝ},
    pts::AbstractVector{P},
    vecs::AbstractVector{T};
    geodesic_interpolation = -1,
    wiresphere = true,
    wires = 32,
    wires_latitude = wires,
    wires_longitude = wires,
    wireframe_color = RGBA(0.0, 0.0, 0.0, 1.0),
    solidsphere = false,
    solid_resolution = 32,
    solid_resolution_latitude = solid_resolution,
    solid_resolution_longitude = solid_resolution,
    solid_color = RGBA(0.9, 0.9, 0.9, 0.8),
) where {P, T}
    # part I: wire
    if wiresphere
        u = range(0, 2π, length = wires_longitude + 1)
        v = range(0, π, length = wires_latitude + 1)
        x = cos.(u) * sin.(v)'
        y = sin.(u) * sin.(v)'
        z = repeat(cos.(v)',outer=[wires_longitude+1, 1])
        @series begin
            seriestype := :wireframe
            color := wireframe_color
            x, y, z
        end
    end
    # part II: solid sphere
    if solidsphere
        u = range(0, 2π, length = solid_resolution_longitude + 1)
        v = range(0, π, length = solid_resolution_latitude + 1)
        x = cos.(u) * sin.(v)'
        y = sin.(u) * sin.(v)'
        z = repeat(cos.(v)',outer=[wires_longitude + 1, 1])
        @series begin
            seriestype := :surface
            color := solid_color
            x, y, z
        end
    end
    show_axis --> false
    framestyle -> :none
    axis --> false
    xlims --> (-1.01, 1.01)
    ylims --> (-1.01, 1.01)
    zlims --> (-1.01, 1.01)
    grid --> false
    aspect_ratio --> :equal
    tickfontcolor --> RGBA(1.0, 1.0, 1.0, 1.0)
    seriestype := :quiver
    quiver := ([v[1] for v in vecs], [v[2] for v in vecs], [v[3] for v in vecs])
    return [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]
end