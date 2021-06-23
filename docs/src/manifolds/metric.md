# Metric manifold

A Riemannian manifold always consists of a [topological manifold](https://en.wikipedia.org/wiki/Topological_manifold) together with a smoothly varying metric $g$.

However, often there is an implicitly assumed (default) metric, like the usual inner product on [`Euclidean`](@ref) space.
This decorator takes this into account.
It is not necessary to use this decorator if you implement just one (or the first) metric.
If you later introduce a second, the old (first) metric can be used with the (non [`MetricManifold`](@ref)) [`AbstractManifold`](@ref), i.e. without an explicitly stated metric.

This manifold decorator serves two purposes:

1. to implement different metrics (e.g. in closed form) for one [`AbstractManifold`](@ref)
2. to provide a way to compute geodesics on manifolds, where this [`AbstractMetric`](@ref) does not yield closed formula.

```@contents
Pages = ["metric.md"]
Depth = 2
```

Note that a metric manifold is an [`AbstractConnectionManifold`](@ref) with the [`LeviCivitaConnection`](@ref) of the metric $g$, and thus a large part of metric manifold's functionality relies on this.

Let's first look at the provided types.

## Types

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/MetricManifold.jl"]
Order = [:type]
```

## Implement Different Metrics on the same Manifold

In order to distinguish different metrics on one manifold, one can introduce two [`AbstractMetric`](@ref)s and use this type to dispatch on the metric, see [`SymmetricPositiveDefinite`](@ref).
To avoid overhead, one [`AbstractMetric`](@ref) can then be marked as being the default, i.e. the one that is used, when no [`MetricManifold`](@ref) decorator is present.
This avoids reimplementation of the first existing metric, access to the metric-dependent functions that were implemented using the undecorated manifold, as well as the transparent fallback of the corresponding [`MetricManifold`](@ref) with default metric to the undecorated implementations.
This does not cause any runtime overhead.
Introducing a default [`AbstractMetric`](@ref) serves a better readability of the code when working with different metrics.

## Implementation of Metrics

For the case that a [`local_metric`](@ref) is implemented as a bilinear form that is positive definite, the following further functions are provided, unless the corresponding [`AbstractMetric`](@ref) is marked as default – then the fallbacks mentioned in the last section are used for e.g. the [`exp!`](@ref)onential map.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/MetricManifold.jl"]
Order = [:function]
```

## Metrics, charts and bases of vector spaces

All metric-related functions take a basis of a vector space as one of the arguments. This needed because generally there is no way to define these functions without referencing a basis. In some cases there is no need to be explicit about this basis, and then for example a [`DefaultOrthonormalBasis`](@ref) object can be used. In cases where being explicit about these bases is needed, for example when using multiple charts, a basis can be specified, for example using [`induced_basis`](@ref Main.Manifolds.induced_basis).

Metric-related functions can take bases of associated tangent spaces as arguments. For example [`local_metric`](@ref) can take the basis of the tangent space it is supposed to operate on instead of a custom basis of the space of symmetric bilinear operators.
