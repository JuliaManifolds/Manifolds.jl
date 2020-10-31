using Manifolds, Plots, RecipesBase

@recipe function f(
    M::Sphere{2},
    pts::AbstractVector{T};
    show_wireframe = true,
    show_sphere = true,
    sphere_color = RGBA(1.,1.,1.,0.4),
    wireframe_color = RGBA(0.0,0.0,0.2,0.4),
    wireframe_lat=33,
    wireframe_lon=33,
    curve_interpolation = -1,
) where {T}
    φ_range = range(0,stop=2*π,length=wireframe_lon)
    λ_range = range(0,stop=π,length=wireframe_lat)
    x = [cos(φ) * sin(λ) for φ ∈ φ_range, λ ∈ λ_range]
    y = [sin(φ) * sin(λ) for φ ∈ φ_range, λ ∈ λ_range]
    z = [cos(λ) for φ ∈ φ_range, λ ∈ λ_range]
    # global options
    scene = plot()
    show_sphere && surface!(
        scene,
        x,y,z,
        color = fill(sphere_color, wireframe_lat, wireframe_lon),
    )
    show_wireframe && wireframe!(
        scene,
        x,
        y,
        z,
        linewidth = 1.2, color = wireframe_color,
    )
    framestyle -> :none
    axis -> false
    xlims -> (-1.01, 1.01)
    ylims -> (-1.01, 1.01)
    zlims -> (-1.01, 1.01)
    if curve_interpolation < 0
        seriestype --> :scatter
        return [p[1] for p ∈ pts], [p[2] for p ∈ pts], [p[3] for p ∈ pts]
    else
        lpts = empty(pts)
        for i=1:(length(pts)-1)
            # push interims points on geodesics between two points.
            push!(lpts,
                shortest_geodesic(
                    M,
                    pts[i],
                    pts[i+1],
                    collect(range(0,1,length=curve_interpolation+2))[1:end-1], # omit end point
                )...
            )
        end
        push!(lpts,last(pts)) # add last end point
        # split into x, y, z and plot as curve
        seriestype --> :path
        return [p[1] for p ∈ lpts], [p[2] for p ∈ lpts], [p[3] for p ∈ lpts]
    end
end

M = Sphere(2)
pts = [ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0] ]
plot(M, pts; curve_interpolation=18,show_axis=false)