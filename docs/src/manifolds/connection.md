# Connection manifold

A connection manifold always consists of a [topological manifold](https://en.wikipedia.org/wiki/Topological_manifold) together with a connection $\Gamma$.

However, often there is an implicitly assumed (default) connection, like the [`LeviCivitaConnection`](@ref) connection on a Riemannian manifold.
It is not necessary to use this decorator if you implement just one (or the first) connection.
If you later introduce a second, the old (first) connection can be used with the (non [`AbstractConnectionManifold`](@ref)) [`AbstractManifold`](@ref), i.e. without an explicitly stated connection.

This manifold decorator serves two purposes:

1. to implement different connections (e.g. in closed form) for one [`AbstractManifold`](@ref)
2. to provide a way to compute geodesics on manifolds, where this [`AbstractMetric`](@ref) does not yield a closed formula.

```@contents
Pages = ["connection.md"]
Depth = 2
```

## Types

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/ConnectionManifold.jl"]
Order = [:type]
```

## Functions

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/ConnectionManifold.jl"]
Order = [:function]
```
