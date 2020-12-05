#
# Defaults
#
CIRCLE_DEFAULT_PLOT_POINTS = 720
WIREFRAME_DEFAULT = 32
SURFACE_RESOLUTION_DEFAULT = 32
#
# Plotting Recipe – Poincaré Ball
#
@recipe function f(
    M::Hyperbolic{2},
    pts::Union{AbstractVector{P},Nothing} = nothing,
    vecs::Union{AbstractVector{T},Nothing} = nothing;
    circle_points = CIRCLE_DEFAULT_PLOT_POINTS,
    geodesic_interpolation = -1,
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
    if pts !== nothing && vecs === nothing
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
    if pts !== nothing && vecs !== nothing
        seriestype := :quiver
        quiver := ([v.value[1] for v in vecs], [v.value[2] for v in vecs])
        return [p.value[1] for p in pts], [p.value[2] for p in pts]
    end
end
#
# Plotting Recipe – Poincaré Half plane
#
@recipe function f(
    M::Hyperbolic{2},
    pts::Union{AbstractVector{P},Nothing}=nothing,
    vecs::Union{AbstractVector{T}}=nothing;
    geodesic_interpolation = -1,
) where {P<:PoincareHalfSpacePoint,T<:PoincareHalfSpaceTVector}
    aspect_ratio --> :equal
    framestyle --> :origin
    seriestype := :quiver
    if pts !== nothing && vecs === nothing
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
    if pts !== nothing && vecs !== nothing
        quiver := ([v.value[1] for v in vecs], [v.value[2] for v in vecs])
        return [p.value[1] for p in pts], [p.value[2] for p in pts]
    end
end
#
# Plotting Recipe – Hyperboloid
#
@recipe function f(
    ::Hyperbolic{2},
    pts::Union{AbstractVector{P},Nothing} = nothing,
    vecs::Union{AbstractVector{T},Nothing} = nothing;
    geodesic_interpolation = -1,
    wiresphere = true,
    wires = WIREFRAME_DEFAULT,
    wires_x = wires,
    wires_y = wires,
    wireframe_color = RGBA(0.0, 0.0, 0.0, 1.0),
    surface = false,
    surface_resolution = SURFACE_RESOLUTION_DEFAULT,
    surface_resolution_x = surface_resolution,
    surface_resolution_y = surface_resolution,
    surface_color = RGBA(0.9, 0.9, 0.9, 0.8),
) where {P,T}
    px = [p[1] for p in pts]
    py = [p[2] for p in pts]
    pz = [p[3] for p in pts]
    # part I: wire
    if wiresphere
        x = range(min(px...), max(px...), length = wires_x)
        y = range(min(py...), max(py...), length = wires_y)
        z = sqrt.(1 .+ (x .^ 2)' .+ y .^ 2)
        @series begin
            seriestype := :wireframe
            seriescolor := wireframe_color
            x, y, z
        end
    end
    # part II: solid sphere
    if surface
        x = range(min(px...), max(px...), length = surface_resolution_x)
        y = range(min(py...), max(py...), length = surface_resolution_y)
        z = sqrt.(1 .+ (x .^ 2)' .+ y .^ 2)
        @series begin
            seriestype := :surface
            color := surface_color
            x, y, z
        end
    end
    show_axis --> false
    framestyle -> :none
    axis --> false
    xlims --> (min(px...), max(px...))
    ylims --> (min(py...), max(py...))
    zlims --> (min(pz...), max(pz...))
    grid --> false
    colorbar --> false
    tickfontcolor --> RGBA(1.0, 1.0, 1.0, 1.0)
    #
    # just pts given -> plot points
    if pts !== nothing && vecs == nothing
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
    if pts !== nothing && vecs !== nothing
        seriestype := :quiver
        quiver := ([v[1] for v in vecs], [v[2] for v in vecs], [v[3] for v in vecs])
        return [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]
    end
end
#
# Plotting Recipe – Sphere
#
@recipe function f(
    M::Sphere{2,ℝ},
    pts::Union{AbstractVector{P},Nothing}=nothing,
    vecs::Union{AbstractVector{T},Nothing}=nothing;
    geodesic_interpolation = -1,
    wiresphere = true,
    wires = WIREFRAME_DEFAULT,
    wires_latitude = wires,
    wires_longitude = wires,
    wireframe_color = RGBA(0.0, 0.0, 0.0, 1.0),
    surface = false,
    surface_resolution = SURFACE_RESOLUTION_DEFAULT,
    surface_resolution_latitude = surface_resolution,
    surface_resolution_longitude = surface_resolution,
    surface_color = RGBA(0.9, 0.9, 0.9, 0.8),
) where {P,T}
    # part I: wire
    if wiresphere
        u = range(0, 2π, length = wires_longitude + 1)
        v = range(0, π, length = wires_latitude + 1)
        x = cos.(u) * sin.(v)'
        y = sin.(u) * sin.(v)'
        z = repeat(cos.(v)', outer = [wires_longitude + 1, 1])
        @series begin
            seriestype := :wireframe
            seriescolor := wireframe_color
            x, y, z
        end
    end
    # part II: solid sphere
    if surface
        u = range(0, 2π, length = surface_resolution_longitude + 1)
        v = range(0, π, length = surface_resolution_latitude + 1)
        x = cos.(u) * sin.(v)'
        y = sin.(u) * sin.(v)'
        z = repeat(cos.(v)', outer = [wires_longitude + 1, 1])
        @series begin
            seriestype := :surface
            color := surface_color
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
    colorbar --> false
    aspect_ratio --> :equal
    tickfontcolor --> RGBA(1.0, 1.0, 1.0, 1.0)
    if pts !== nothing && vecs == nothing #plot points/geodesics
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
    if pts !== nothing && vecs !== nothing # plot tangents/quiver
        seriestype := :quiver
        quiver := ([v[1] for v in vecs], [v[2] for v in vecs], [v[3] for v in vecs])
        return [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]
    end
end