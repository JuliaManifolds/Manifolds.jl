# Sphere and unit norm arrays

```@docs
AbstractSphere
```

The classical sphere, i.e. unit norm (real- or complex-valued) vectors can be generated as usual: to create the 2-dimensional sphere (in $ℝ^3$), use `Sphere(2)` and `Sphere(2,ℂ)`, respectively.

```@docs
Sphere
```

For the higher-dimensional arrays, for example unit (Frobenius) norm matrices, the manifold is generated using the size of the matrix.
To create the unit sphere of $3×2$ real-valued matrices, write `ArraySphere(3,2)` and the complex case is done – as for the [`Euclidean`](@ref) case – with an keyword argument `ArraySphere(3,2; field = ℂ)`. This case also covers the classical sphere as a special case, but you specify the size of the vectors/embedding instead: The 2-sphere can here be generated `ArraySphere(3)`.

```@docs
ArraySphere
```

## Functions on unit spheres
```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Sphere.jl"]
Order = [:function]
```

## Visualization on `Sphere{2,ℝ}`
You can visualize both points and tangent vectors on the sphere.

!!! note
    There seems to be no unified way to draw spheres in the backends of [Plots.jl](http://docs.juliaplots.org/latest/).
    This recipe currently uses the `seriestype` `wireframe` and `surface`, which does not yet work with the default backend [`GR`](https://github.com/jheinen/GR.jl).

In general you can plot the surface of the hyperboloid either as wireframe (`wireframe=true`) additionally specifying `wires` (or `wires_x` and `wires_y`) to change the density of the wires and a `wireframe_color` for their color. The same holds for the plot as a `surface` (which is `false` by default) and its `surface_resolution` (or `surface_resolution_lat` or `surface_resolution_lon`) and a `surface_color`.

```@example sphereplot1
using Manifolds, Plots
pyplot()
M = Sphere(2)
pts = [ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0] ]
scene = plot(M, pts; wireframe_color=colorant"#CCCCCC", markersize=10)
```

which scatters our points. We can also draw connecting geodesics, which here is a geodesic triangle. Here we discretize each geodesic with 100 points along the geodesic.
The default value is `geodesic_interpolation=-1` which switches to scatter plot of the data.

```@example sphereplot1
plot!(scene, M, pts; wireframe=false, geodesic_interpolation=100, linewidth=2)
```

And we can also add tangent vectors, for example tangents pointing towards the geometric center of given points.

```@example sphereplot1
pts2 =  [ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0] ]
p3 = 1/sqrt(3) .* [1.0, -1.0, 1.0]
vecs = log.(Ref(M), pts2, Ref(p3))
plot!(scene, M, pts2, vecs; wireframe = false, linewidth=1.5)
```
