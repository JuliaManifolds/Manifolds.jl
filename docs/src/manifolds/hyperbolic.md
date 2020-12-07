# Hyperbolic space

The hyperbolic space can be represented in three different models.

* [Hyperboloid](@ref hyperboloid_model) which is the default model, i.e. is used when using arbitrary array types for points and tangent vectors
* [Poincaré ball](@ref poincare_ball) with separate types for points and tangent vectors and a [visualization](@ref poincare_ball_plot) for the two-dimensional case
* [Poincaré half space](@ref poincare_halfspace) with separate types for points and tangent vectors and a [visualization](@ref poincare_half_plane_plot) for the two-dimensional cae.

In the following the common functions are collected.

A function in this general section uses vectors interpreted as if in the [hyperboloid model](@ref hyperboloid_model),
and other representations usually just convert to this representation to use these general functions.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Hyperbolic.jl"]
Order = [:type, :function]
```

## [hyperboloid model](@id hyperboloid_model)

```@autodocs
Modules = [Manifolds]
Private = false
Pages = ["manifolds/HyperbolicHyperboloid.jl"]
Order = [:type, :function]
```

### [Visualization of the Hyperboloid](@id hyperboloid_plot)

For the case of [`Hyperbolic`](@ref)`(2)` there is a plotting available based on a [PlottingRecipe](https://docs.juliaplots.org/latest/recipes/) you can easily plot points, connecting geodesics as well as tangent vectors.

!!! note
    The recipes are only loaded if [Plots.jl](http://docs.juliaplots.org/latest/) or
    [RecipesBase.jl](http://juliaplots.org/RecipesBase.jl/stable/) is loaded.

If we consider a set of points, we can first plot these and their connecting
geodesics using the `geodesic_interpolation` For the points. This variable specifies with how many points a geodesic between two successive points is sampled (per default it's `-1`, which deactivates geodesics) and the line style is set to be a path.

In general you can plot the surface of the hyperboloid either as wireframe (`wireframe=true`) additionally specifying `wires` (or `wires_x` and `wires_y`) for more precision and a `wireframe_color`. The same holds for the plot as a `surface` (which is `false` by default) and its `surface_resolution` (or `surface_resolution_x` or `surface_resolution_y`) and a `surface_color`.

```@example hyperboloid
using Manifolds, Plots
M = Hyperbolic(2)
pts =  [ [0.85*cos(φ), 0.85*sin(φ), sqrt(0.85^2+1)] for φ ∈ range(0,2π,length=11) ]
scene = plot(M, pts; geodesic_interpolation=100)
```

To just plot the points atop, we can just omit the `geodesic_interpolation` parameter to obtain a scatter plot. NOte that we avoid redrawing the wireframe in the following `plot!`s.

```@example hyperboloid
plot!(scene, M, pts; wireframe=false)
```

We can further generate tangent vectors in these spaces and use a plot for there. Keep in mind, that a tangent vector in plotting always requires its base point

```@example hyperboloid
pts2 = [ [0.45 .*cos(φ + 6π/11), 0.45 .*sin(φ + 6π/11), sqrt(0.45^2+1) ] for φ ∈ range(0,2π,length=11)]
vecs = log.(Ref(M),pts,pts2)
plot!(scene, M, pts, vecs; wireframe=false)
```

Just to illustrate, for the first point the tangent vector is pointing along the following geodesic

```@example hyperboloid
plot!(scene, M, [pts[1], pts2[1]]; geodesic_interpolation=100, wireframe=false)
```

### Internal functions

The following functions are available for internal use to construct points in the hyperboloid model

```@autodocs
Modules = [Manifolds]
Public = false
Pages = ["manifolds/HyperbolicHyperboloid.jl"]
Order = [:type, :function]
```

## [Poincaré ball model](@id poincare_ball)

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/HyperbolicPoincareBall.jl"]
Order = [:type, :function]
```

### [Visualization of the Poincaré ball](@id poincare_ball_plot)

For the case of [`Hyperbolic`](@ref)`(2)` there is a plotting available based on a [PlottingRecipe](https://docs.juliaplots.org/latest/recipes/) you can easily plot points, connecting geodesics as well as tangent vectors.

!!! note
    The recipes are only loaded if [Plots.jl](http://docs.juliaplots.org/latest/) or
    [RecipesBase.jl](http://juliaplots.org/RecipesBase.jl/stable/) is loaded.

If we consider a set of points, we can first plot these and their connecting
geodesics using the `geodesic_interpolation` For the points. This variable specifies with how many points a geodesic between two successive points is sampled (per default it's `-1`, which deactivates geodesics) and the line style is set to be a path.
Another keyword argument added is the border of the Poincaré disc, namely
`circle_points = 720` resolution of the drawn boundary (every hlaf angle) as well as its color, `hyperbolic_border_color = RGBA(0.0, 0.0, 0.0, 1.0)`.

```@example poincareball
using Manifolds, Plots
M = Hyperbolic(2)
pts = PoincareBallPoint.( [0.85 .* [cos(φ), sin(φ)] for φ ∈ range(0,2π,length=11)])
scene = plot(M, pts, geodesic_interpolation = 100)
```

To just plot the points atop, we can just omit the `geodesic_interpolation` parameter to obtain a scatter plot

```@example poincareball
plot!(scene, M, pts)
```

We can further generate tangent vectors in these spaces and use a plot for there. Keep in mind, that a tangent vector in plotting always requires its base point

```@example poincareball
pts2 = PoincareBallPoint.( [0.45 .* [cos(φ + 6π/11), sin(φ + 6π/11)] for φ ∈ range(0,2π,length=11)])
vecs = log.(Ref(M),pts,pts2)
plot!(scene, M, pts,vecs)
```

Just to illustrate, for the first point the tangent vector is pointing along the following geodesic

```@example poincareball
plot!(scene, M, [pts[1], pts2[1]], geodesic_interpolation=100)
```

## [Poincaré half space model](@id poincare_halfspace)

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/HyperbolicPoincareHalfspace.jl"]
Order = [:type, :function]
```

### [Visualization on the Poincaré half plane](@id poincare_half_plane_plot)

For the case of [`Hyperbolic`](@ref)`(2)` there is a plotting available based on a [PlottingRecipe](https://docs.juliaplots.org/latest/recipes/) you can easily plot points, connecting geodesics as well as tangent vectors.

!!! note
    The recipes are only loaded if [Plots.jl](http://docs.juliaplots.org/latest/) or
    [RecipesBase.jl](http://juliaplots.org/RecipesBase.jl/stable/) is loaded.

We again have two different recipes, one for points, one for tangent vectors, where the first one again can be equipped with geodesics between the points.
In the following example we generate 7 points on an ellipse in the [Hyperboloid model](#hyperboloid-model).

```@example poincarehalfplane
using Manifolds, Plots
M = Hyperbolic(2)
pre_pts = [2.0 .* [5.0*cos(φ), sin(φ)] for φ ∈ range(0,2π,length=7)]
pts = convert.(
    Ref(PoincareHalfSpacePoint),
    Manifolds._hyperbolize.(Ref(M), pre_pts)
)
scene = plot(M, pts, geodesic_interpolation = 100)
```

To just plot the points atop, we can just omit the `geodesic_interpolation` parameter to obtain a scatter plot

```@example poincarehalfplane
plot!(scene, M, pts)
```

We can further generate tangent vectors in these spaces and use a plot for there. Keep in mind, that a tangent vector in plotting always requires its base point.
Here we would like to look at the tangent vectors pointing to the `origin`

```@example poincarehalfplane
origin = PoincareHalfSpacePoint([0.0,1.0])
vecs = [log(M,p,origin) for p ∈ pts]
scene = plot!(scene, M, pts, vecs)
```

And we can again look at the corresponding geodesics, for example

```@example poincarehalfplane
plot!(scene, M, [pts[1], origin], geodesic_interpolation=100)
plot!(scene, M, [pts[2], origin], geodesic_interpolation=100)
```
