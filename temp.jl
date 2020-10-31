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
    x = zeros(wireframe_lat,wireframe_lon); y = deepcopy(x); z = deepcopy(y)
    for (i,lon) ∈ enumerate(range(0,stop=2*π,length=wireframe_lat))
        for (j,lat) ∈ enumerate(range(0,stop=π,length=wireframe_lon))
            @inbounds x[i,j] = cos(lon) * sin(lat)
            @inbounds y[i,j] = sin(lon) * sin(lat)
            @inbounds z[i,j] = cos(lat)
        end
    end
    # global options
    options = []
    scene = surface(; options...)
    show_sphere && surface!(
        scene,
        x,y,z,
        color = fill(sphere_color, wireframe_lat, wireframe_lon),
        options...
    )
    show_wireframe && wireframe!(
        scene,
        x,
        y,
        z,
        linewidth = 1.2, color = wireframe_color,
        options...
    )
    (curve_interpolation < 0) && scatter!(
        scene,
        pts,
        options...
    )
    if curve_interpolation >= 0
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
        plot!(scene, [p[1], p ∈ lpts], [p[2], p ∈ lpts], [p[3], p ∈ lpts])
    end
    return scene
end

M = Sphere(2)
pts = [ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0] ]
plot(M, pts)