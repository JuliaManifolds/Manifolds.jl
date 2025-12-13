# Metric manifold

A Riemannian manifold always consists of a [topological manifold](https://en.wikipedia.org/wiki/Topological_manifold) together with a smoothly varying metric tensor ``g``.
This metric tensor defines an inner product on each tangent space of the manifold.
In `Manifolds.jl`, a concrete implementation of an [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) has an implicit metric, which we refer to as the default metric. For example on [`Euclidean`](@ref) space, this is the usual inner product.
For the [`Sphere`](@ref), this default metric is the metric induced by the embedding in the ambient Euclidean space, restricted to the tangent spaces of the sphere.

Such a default metric is usually not the only possible metric on a manifold.

The [`MetricManifold`](@extref ManifoldsBase.MetricManifold) decorator allows to introduce further metrics for a manifold by acting as a wrapper.
This allows to define further metrics for an existing manifold, while only having to implement functions that depend on the metric.
All functions that are independent of the metric are automatically forwarded to the underlying manifold.
When wrapping a manifold `M` with a [`MetricManifold`](@extref ManifoldsBase.MetricManifold) together with the default [`metric`](@extref ManifoldsBase.metric)`(M)`,
this wrapper acts completely transparent and passes all function calls to the underlying manifold `M`.

```@contents
Pages = ["metric.md"]
Depth = 2
```

## Deprecated methods, to be replaced by chart-based variants

```@docs
einstein_tensor(::AbstractManifold, ::Any, ::AbstractBasis)
ricci_curvature(::AbstractManifold, ::Any, ::AbstractBasis)
local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
local_metric_jacobian(::AbstractManifold, ::Any, ::AbstractBasis, ::AbstractDiffBackend)
inverse_local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
log_local_metric_density(::AbstractManifold, ::Any, ::AbstractBasis)
det_local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
```

## Types

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/MetricManifold.jl"]
Order = [:type]
```

## Implement Different Metrics on the same Manifold

In order to distinguish different metrics on one manifold, one can introduce two [`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`)s and use this type to dispatch on the metric, see [`SymmetricPositiveDefinite`](@ref).
To avoid overhead, one [`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) can then be marked as being the default, i.e. the one that is used, when no [`MetricManifold`](@extref ManifoldsBase.MetricManifold) decorator is present.
This avoids reimplementation of the first existing metric, access to the metric-dependent functions that were implemented using the undecorated manifold, as well as the transparent fallback of the corresponding [`MetricManifold`](@extref ManifoldsBase.MetricManifold) with default metric to the undecorated implementations.
This does not cause any runtime overhead.
Introducing a default [`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) serves a better readability of the code when working with different metrics.

## Implementation of Metrics

For the case that a [`local_metric`](@ref) is implemented as a bilinear form that is positive definite, the following further functions are provided, unless the corresponding [`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) is marked as default – then the fallbacks mentioned in the last section are used for e.g. the exponential map.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/MetricManifold.jl"]
Order = [:function]
```

## Metrics, charts and bases of vector spaces

Metric-related functions, similarly to connection-related functions, need to operate in a basis of a vector space, see [here](@ref connections_charts).

Metric-related functions can take bases of associated tangent spaces as arguments. For example [`local_metric`](@ref) can take the basis of the tangent space it is supposed to operate on instead of a custom basis of the space of symmetric bilinear operators.
