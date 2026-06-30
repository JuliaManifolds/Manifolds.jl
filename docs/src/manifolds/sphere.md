# [Sphere and unit norm arrays](@id SphereSection)

```@docs
AbstractSphere
```

The classical sphere, i.e. unit norm (real- or complex-valued) vectors can be generated as usual: to create the 2-dimensional sphere (in ``ℝ^3``), use `Sphere(2)` and `Sphere(2,ℂ)`, respectively.

```@docs
Sphere
```

For the higher-dimensional arrays, for example unit (Frobenius) norm matrices, the manifold is generated using the size of the matrix.
To create the unit sphere of ``3×2`` real-valued matrices, write `ArraySphere(3,2)` and the complex case is done – as for the [`Euclidean`](@ref) case – with an keyword argument `ArraySphere(3,2; field=ℂ)`. This case also covers the classical sphere as a special case, but you specify the size of the vectors/embedding instead: The 2-sphere can here be generated `ArraySphere(3)`.

```@docs
ArraySphere
```

There is also one atlas available on the sphere.

```@docs
Manifolds.StereographicAtlas
```

## Functions on unit spheres
```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Sphere.jl"]
Order = [:function]
```

## Visualization of data on spheres

For the 2-sphere and the 1-sphere, i.e. data on the embedded Circle on the plane,
data can be visualized using [`ManifoldMakie.jl`](https://juliamanifolds.github.io/ManifoldMakie.jl/stable/), see [here](https://juliamanifolds.github.io/ManifoldMakie.jl/stable/sphere/).

`Manifold.jl` has a legacy extension to [`Plots.jl`](http://docs.juliaplots.org/latest/)
using the recipes for `seriestype` `wireframe` and `surface`.
Since the backends for [`Plots.jl`](http://docs.juliaplots.org/latest/) are not consistent, this only works some backends and for example not with their default backend [`GR.jl`](https://github.com/jheinen/GR.jl).

## Literature

```@bibliography
Pages = ["sphere.md"]
Canonical=false
```