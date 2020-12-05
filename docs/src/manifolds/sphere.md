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

```@example sphereplot1
using Manifolds Plots
M = Sphere(2)
pts = [ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0] ]
scene = plot(M,pts)
```

which scatters our points. We can also draw connecting geodesics, which here is a geodesic triangle. Here we discretise each geodesic with 100 points along the geodesic.

```@example sphereplot1
plot!(scene, M, pts; geodesic_interpoaltion=100)
```

And we can also add tangent vectors, for example tangents pointing to the middle

```@example sphereplot1
pts2 =  [ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0] ]
p3 = 1/sqrt(3) .* [1.0, 1.0, 1.0]
vecs = log.(Ref(M), pts2, Ref(p3))
plot!(scene, M, pts2, vecs)
```

!!! note
    The recipes are only loaded if [Plots.jl](http://docs.juliaplots.org/latest/) or
    [RecipesBase.jl](http://juliaplots.org/RecipesBase.jl/stable/) is loaded.
    Furthermore, the `surface` does not yet work in the `GR` backend.
