# Hyperbolic space

The hyperbolic space can be represented in three different ways. In the following the common functions are collected.
Then, specific methods for each of the representations, especially conversions between those.

A function in this general section uses vectors interpreted as if in the [hyperboloid model](@ref),
and other representations usually just convert to this representation to use these general functions.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Hyperbolic.jl"]
Order = [:type, :function]
```

## hyperboloid model

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/HyperbolicHyperboloid.jl"]
Order = [:type, :function]
```

## Poincaré ball model

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/HyperbolicPoincareBall.jl"]
Order = [:type, :function]
```

### Plotting
For the case of [`Hyperbolic`](@ref)`(2)` there is a plotting available based on a [PlottingRecipe](https://docs.juliaplots.org/latest/recipes/) you can easily plot points, connecting geodesics as well as tangent vectors.

If we consider a set of points, we can first plot these and their connecting
geodeics using the `geodesic_interpolation` For the points. This variable specifies with how many points a geodesic between two successive points is sampled (per default it's `-1`, which deactivates geodesics) and the line style is set to be a path.

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

## Poincaré half space model

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/HyperbolicPoincareHalfspace.jl"]
Order = [:type, :function]
```

### Plotting
For the case of [`Hyperbolic`](@ref)`(2)` there is a plotting available based on a [PlottingRecipe](https://docs.juliaplots.org/latest/recipes/) you can easily plot points, connecting geodesics as well as tangent vectors.
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
