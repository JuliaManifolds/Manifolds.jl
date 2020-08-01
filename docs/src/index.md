# Manifolds

The package __Manifolds__ aims to provide a library of manifolds to be used within your project.
The implemented manifolds are accompanied by their mathematical formulae.

The manifolds are implemented using the interface for manifolds given in [`ManifoldsBase.jl`](interface.md).
You can use that interface to implement your own software on manifolds, such that all manifolds
based on that interface can be used within your code.

For more information, see the [About](misc/about.md) section.

## Getting started

To install the package just type

```julia
] add Manifolds
```

Then you can directly start, for example to stop half way from the north pole on the [`Sphere`](@ref) to a point on the the equator, you can generate the [`shortest_geodesic`](@ref).
It internally employs [`log`](@ref log(::Sphere,::Any,::Any)) and [`exp`](@ref exp(::Sphere,::Any,::Any)).

```@example
using Manifolds
M = Sphere(2)
γ = shortest_geodesic(M, [0., 0., 1.], [0., 1., 0.])
γ(0.5)
```
