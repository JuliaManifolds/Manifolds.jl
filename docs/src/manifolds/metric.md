# Metric manifold

A Riemannian manifold always consists of a [topological manifold](https://en.wikipedia.org/wiki/Topological_manifold) together with a smoothly varying metric $g$.

However, often there is a implicitly assumed (daefault) metric, like the usual
inner product on [`Euclidean`](@ref) space. This decorator takes this into
account. Hence it is not necessary to acts as a (as transparent as possible)
decorator, see [`is_decorator_manifold`](@ref) for details.

This manifold servers two purposes: To implement different metrics (i.g. in closed
form) for one [`Manifold`](@ref) and to provide a way to compute geodesics
on manifolds, where this [`Metric`](@ref) does not yield closed formula. 

Let's first look at the provided types.

## types

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
[`MetricManifold`](@ref) is used. This avoids reimplementation of a first
existing metric, access for the default metric functions (without an [`MetricManifold`](@ref)
overhead) as well as the transparent fallback of the corresponding [`MetricManifold`](@ref)
with default metric to the (classical) implementations.

## Implementaion of Metrics

For the case that a [`local_metric`](@ref) is implemented as a bilinear form
that is positive definite, the following further functions are provided,
unless the corresponding [`Metric`](@ref) is marked as default – then the fallbacks
mentioned in the last section are used for e.g. the [`exp!`](@ref)onential map.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["Metric.jl"]
Order = [:function]
```
