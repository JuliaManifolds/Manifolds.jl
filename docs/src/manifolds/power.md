# Power manifold

A power manifold is based on a [`Manifold`](@ref) $\mathcal M$ to build a
$\mathcal M^{n_1 \times n_2 \times \cdots \times n_m}$.
For the two special cases $m=1$ and $m=2$ we obtain
one-dimensional data, i.e. vectors of points on $\mathcal M$ and matrices of images,
respectively.

## Example
There are two ways to store the data: in a multidimensional array or in a nested array.

Let's look at both for an example. Let $\mathcal M$ be `Sphere(2)` the 2-sphere. and we want
to look at vectors of length 4.

For the default, the [`ArrayPowerRepresentation`](@ref), we store the data in a multidimensional array,

```@example 1
using Manifolds #hide
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

An advantage of this representation is that it is quite efficient, especially when a `HybridArray` (from the `HybridArrays.jl` package) is used to represent a point on the power manifold.
A disadvantage is not being able to easily identify parts of the multidimensional array that correspond to a single point on the base manifold.
Another problem is, that accessing a single point is ` p[:,1]` which might be unintuitive.

For the [`NestedPowerRepresentation`](@ref) we can now do

```@example 2
using Manifolds #hide
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
