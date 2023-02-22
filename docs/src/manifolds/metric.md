# Metric manifold

A Riemannian manifold always consists of a [topological manifold](https://en.wikipedia.org/wiki/Topological_manifold) together with a smoothly varying metric $g$.

However, often there is an implicitly assumed (default) metric, like the usual inner product on [`Euclidean`](@ref) space.
This decorator takes this into account.
It is not necessary to use this decorator if you implement just one (or the first) metric.
If you later introduce a second, the old (first) metric can be used with the (non [`MetricManifold`](@ref)) `AbstractManifold`, i.e. without an explicitly stated metric.

This manifold decorator serves two purposes:

1. to implement different metrics (e.g. in closed form) for one `AbstractManifold`
2. to provide a way to compute geodesics on manifolds, where this [`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) does not yield closed formula.

```@contents
Pages = ["metric.md"]
Depth = 2
```

Note that a metric manifold is has a [`IsConnectionManifold`](@ref) trait referring to the [`LeviCivitaConnection`](@ref) of the metric $g$, and thus a large part of metric manifold's functionality relies on this.

Let's first look at the provided types.

## Types

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/MetricManifold.jl"]
Order = [:type]
```

## Implement Different Metrics on the same Manifold

In order to distinguish different metrics on one manifold, one can introduce two [`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric)s and use this type to dispatch on the metric, see [`SymmetricPositiveDefinite`](@ref).
To avoid overhead, one [`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) can then be marked as being the default, i.e. the one that is used, when no [`MetricManifold`](@ref) decorator is present.
This avoids reimplementation of the first existing metric, access to the metric-dependent functions that were implemented using the undecorated manifold, as well as the transparent fallback of the corresponding [`MetricManifold`](@ref) with default metric to the undecorated implementations.
This does not cause any runtime overhead.
Introducing a default [`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) serves a better readability of the code when working with different metrics.

## Implementation of Metrics

For the case that a [`local_metric`](@ref) is implemented as a bilinear form that is positive definite, the following further functions are provided, unless the corresponding [`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) is marked as default – then the fallbacks mentioned in the last section are used for e.g. the exponential map.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/MetricManifold.jl"]
Order = [:function]
```

## Metrics, charts and bases of vector spaces

Metric-related functions, similarly to connection-related functions, need to operate in a basis of a vector space, see [here](@ref connections_charts).

Metric-related functions can take bases of associated tangent spaces as arguments. For example [`local_metric`](@ref) can take the basis of the tangent space it is supposed to operate on instead of a custom basis of the space of symmetric bilinear operators.
