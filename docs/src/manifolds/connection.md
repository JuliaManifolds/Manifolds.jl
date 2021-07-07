# Connection manifold

A connection manifold always consists of a [topological manifold](https://en.wikipedia.org/wiki/Topological_manifold) together with a [connection](https://en.wikipedia.org/wiki/Connection_(mathematics)) $\Gamma$.

However, often there is an implicitly assumed (default) connection, like the [`LeviCivitaConnection`](@ref) connection on a Riemannian manifold.
It is not necessary to use this decorator if you implement just one (or the first) connection.
If you later introduce a second, the old (first) connection can be used with the (non [`AbstractConnectionManifold`](@ref)) [`AbstractManifold`](@ref), i.e. without an explicitly stated connection.

This manifold decorator serves two purposes:

1. to implement different connections (e.g. in closed form) for one [`AbstractManifold`](@ref)
2. to provide a way to compute geodesics on manifolds, where this [`AbstractAffineConnection`](@ref) does not yield a closed formula.

An example of usage can be found in Cartan-Schouten connections, see [`AbstractCartanSchoutenConnection`](@ref).

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

## [Charts and bases of vector spaces](@id connections_charts)

All connection-related functions take a basis of a vector space as one of the arguments. This is needed because generally there is no way to define these functions without referencing a basis. In some cases there is no need to be explicit about this basis, and then for example a [`DefaultOrthonormalBasis`](@ref) object can be used. In cases where being explicit about these bases is needed, for example when using multiple charts, a basis can be specified, for example using [`induced_basis`](@ref Main.Manifolds.induced_basis).
