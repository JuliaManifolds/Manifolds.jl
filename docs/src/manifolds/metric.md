# Metric manifold

A Riemannian manifold always consists of a [topological manifold](https://en.wikipedia.org/wiki/Topological_manifold) together with a smoothly varying metric $g$.

However, often there is an implicitly assumed (default) metric, like the usual
inner product on [`Euclidean`](@ref) space. This decorator takes this into
account. It is not necessary to use this decorator, if you imlement just one (or
the first) metric. If you later introduce a second, the old (first) one can be
used with or without a metricx. The decorator acts transparent in that sense;
see [`is_decorator_manifold`](@ref) for details.

This manifold decorator servers two purposes: to implement different metrics
(e.g. in closed form) for one [`Manifold`](@ref) and to provide a way to compute
geodesics on manifolds, where this [`Metric`](@ref) does not yield closed formula. 

Let's first look at the provided types.

## Types

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["Metric.jl"]
Order = [:type]
```

## Implement Different Metrics on the same Manifold

In order to distinguish different metrics on one manifold, one can introduce
two [`Metric`](@ref)s and use this type to dispatch on the metric, see 
[`SymmetricPositiveDefinite`](@ref). To avoid overhead, one [`Metric`](@ref)
can then be marked as being the default, i.e. the one that is used, when no
[`MetricManifold`](@ref) decorator is present. This avoids reimplementation of the first
existing metric, access for the default metric functions (without an [`MetricManifold`](@ref)
overhead) as well as the transparent fallback of the corresponding [`MetricManifold`](@ref)
with default metric to the undecorated implementations.

## Implementation of Metrics

For the case that a [`local_metric`](@ref) is implemented as a bilinear form
that is positive definite, the following further functions are provided,
unless the corresponding [`Metric`](@ref) is marked as default – then the fallbacks
mentioned in the last section are used for e.g. the [`exp!`](@ref)onential map.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["Metric.jl"]
Order = [:function]
```
