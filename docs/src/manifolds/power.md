# Power manifold

A power manifold is based on a [`Manifold`](@ref) $\mathcal M$ to build a $\mathcal M^{n_1 \times n_2 \times \cdots \times n_m}$.
In the case where $m=1$ we can represent a manifold-valued vector of data of length $n_1$, for example a time series.
The case where $m=2$ is useful for representing manifold-valued matrices of data of size $n_1 \times n_2$, for example certain types of images.

## Example

There are two ways to store the data: in a multidimensional array or in a nested array.

Let's look at an example for both.
Let $\mathcal M$ be `Sphere(2)` the 2-sphere and we want to look at vectors of length 4.

For the default, the [`ArrayPowerRepresentation`](@ref), we store the data in a multidimensional array,

```@example 1
using Manifolds
M = PowerManifold(Sphere(2), 4)
p = cat([1.0, 0.0, 0.0],
        [1/sqrt(2.0), 1/sqrt(2.0), 0.0],
        [1/sqrt(2.0), 0.0, 1/sqrt(2.0)],
        [0.0, 1.0, 0.0]
    ,dims=2)
```

which is a valid point i.e.

```@example 1
is_manifold_point(M, p)
```

This can also be used in combination with [HybridArrays.jl](https://github.com/mateuszbaran/HybridArrays.jl) and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), by setting

```@example 1
using HybridArrays, StaticArrays
q = HybridArray{Tuple{3,StaticArrays.Dynamic()}, Float64, 2}(p)
```

which is still a valid point on `M` and [`PowerManifold`](@ref) works with these, too.

An advantage of this representation is that it is quite efficient, especially when a `HybridArray` (from the [HybridArrays.jl](https://github.com/mateuszbaran/HybridArrays.jl) package) is used to represent a point on the power manifold.
A disadvantage is not being able to easily identify parts of the multidimensional array that correspond to a single point on the base manifold.
Another problem is, that accessing a single point is ` p[:,1]` which might be unintuitive.

For the [`NestedPowerRepresentation`](@ref) we can now do

```@example 2
using Manifolds
M = PowerManifold(Sphere(2), NestedPowerRepresentation(), 4)
p = [ [1.0, 0.0, 0.0],
      [1/sqrt(2.0), 1/sqrt(2.0), 0.0],
      [1/sqrt(2.0), 0.0, 1/sqrt(2.0)],
      [0.0, 1.0, 0.0]
    ]
```

which is again a valid point so [`is_manifold_point`](@ref)`(M,p)` here also yields true.
A disadvantage might be that with nested arrays one loses a little bit of performance.
The data however is nicely encapsulated. Accessing the first data item is just `p[1]`.

## Types and Functions
```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/PowerManifold.jl"]
Order = [:type, :function]
```
