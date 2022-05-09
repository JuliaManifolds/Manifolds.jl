# Power manifold

A power manifold is based on a [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  $\mathcal M$ to build a $\mathcal M^{n_1 \times n_2 \times \cdots \times n_m}$.
In the case where $m=1$ we can represent a manifold-valued vector of data of length $n_1$, for example a time series.
The case where $m=2$ is useful for representing manifold-valued matrices of data of size $n_1 \times n_2$, for example certain types of images.

There are three available representations for points and vectors on a power manifold:

* [`ArrayPowerRepresentation`](@ref) (the default one), very efficient but only applicable when points on the underlying manifold are represented using plain `AbstractArray`s.
* [`NestedPowerRepresentation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.NestedPowerRepresentation), applicable to any manifold. It assumes that points on the underlying manifold are represented using mutable data types.
* [`NestedReplacingPowerRepresentation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.NestedReplacingPowerRepresentation), applicable to any manifold. It does not mutate points on the underlying manifold, replacing them instead when appropriate.

Below are some examples of usage of these representations.

## Example

There are two ways to store the data: in a multidimensional array or in a nested array.

Let's look at an example for both.
Let $\mathcal M$ be `Sphere(2)` the 2-sphere and we want to look at vectors of length 4.

### `ArrayPowerRepresentation`

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
is_point(M, p)
```

This can also be used in combination with [HybridArrays.jl](https://github.com/mateuszbaran/HybridArrays.jl) and [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), by setting

```@example 1
using HybridArrays, StaticArrays
q = HybridArray{Tuple{3,StaticArrays.Dynamic()},Float64,2}(p)
```

which is still a valid point on `M` and [`PowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.PowerManifold) works with these, too.

An advantage of this representation is that it is quite efficient, especially when a `HybridArray` (from the [HybridArrays.jl](https://github.com/mateuszbaran/HybridArrays.jl) package) is used to represent a point on the power manifold.
A disadvantage is not being able to easily identify parts of the multidimensional array that correspond to a single point on the base manifold.
Another problem is, that accessing a single point is ` p[:, 1]` which might be unintuitive.

### `NestedPowerRepresentation`

For the [`NestedPowerRepresentation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.NestedPowerRepresentation) we can now do

```@example 2
using Manifolds
M = PowerManifold(Sphere(2), NestedPowerRepresentation(), 4)
p = [ [1.0, 0.0, 0.0],
      [1/sqrt(2.0), 1/sqrt(2.0), 0.0],
      [1/sqrt(2.0), 0.0, 1/sqrt(2.0)],
      [0.0, 1.0, 0.0],
    ]
```

which is again a valid point so `is_point(M, p)` here also yields true.
A disadvantage might be that with nested arrays one loses a little bit of performance.
The data however is nicely encapsulated. Accessing the first data item is just `p[1]`.

For accessing points on power manifolds in both representations you can use [`get_component`](@ref) and [`set_component!`](@ref) functions.
They work work both point representations.

```@example 3
using Manifolds
M = PowerManifold(Sphere(2), NestedPowerRepresentation(), 4)
p = [ [1.0, 0.0, 0.0],
      [1/sqrt(2.0), 1/sqrt(2.0), 0.0],
      [1/sqrt(2.0), 0.0, 1/sqrt(2.0)],
      [0.0, 1.0, 0.0],
    ]
set_component!(M, p, [0.0, 0.0, 1.0], 4)
get_component(M, p, 4)
```

### `NestedReplacingPowerRepresentation`

The final representation is the [`NestedReplacingPowerRepresentation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.NestedReplacingPowerRepresentation). It is similar to the [`NestedPowerRepresentation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.NestedPowerRepresentation) but it does not perform mutating operations on the points on the underlying manifold. The example below uses this representation to store points on a power manifold of the [`SpecialEuclidean`](@ref) group in-line in an `Vector` for improved efficiency. When having a mixture of both, i.e. an array structure that is nested (like [´NestedPowerRepresentation](@ref)) in the sense that the elements of the main vector are immutable, then changing the elements can not be done in a mutating way and hence [`NestedReplacingPowerRepresentation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.NestedReplacingPowerRepresentation) has to be used.

```@example 4
using Manifolds, StaticArrays
R2 = Rotations(2)

G = SpecialEuclidean(2)
N = 5
GN = PowerManifold(G, NestedReplacingPowerRepresentation(), N)

q = [1.0 0.0; 0.0 1.0]
p1 = [ProductRepr(SVector{2,Float64}([i - 0.1, -i]), SMatrix{2,2,Float64}(exp(R2, q, hat(R2, q, i)))) for i in 1:N]
p2 = [ProductRepr(SVector{2,Float64}([i - 0.1, -i]), SMatrix{2,2,Float64}(exp(R2, q, hat(R2, q, -i)))) for i in 1:N]

X = similar(p1);

log!(GN, X, p1, p2)
```

## Types and Functions
```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/PowerManifold.jl"]
Order = [:type, :function]
```
