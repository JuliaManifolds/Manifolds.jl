# Sphere and unit norm arrays

```@docs
AbstractSphere
```

The classical sphere, i.e. unit norm (real- or complex-valued) vectors can be generated as usual: to create the 2-dimensional sphere (in $ℝ^3$), use `Sphere(2)` and `Sphere(2,ℂ)`, respectively.

```@docs
Sphere
```

For the higher-dimensional arrays, for example unit norm matrices, the manifold is generated using the size of the matrix.
To create the unit sphere of $3×2$ real-valued matrices, write `ArraySphere(3,2)` and the complex case is done – as for the [`Euclidean`](@ref) case – with an keyword argument `ArraySphere(3,2; field = ℂ)`. This case also covers the classical sphere as a special case, but you specify the size of the vectors/embedding instead: The 2-sphere can here be generated `ArraySphere(3)`.

```@docs
ArraySphere
```

# Functions on unit spheres
```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Sphere.jl"]
Order = [:function]
```
