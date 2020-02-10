# Embedded manifold

A lot of manifolds can easily be defined in their embedding. For example the
[`Sphere`](@ref)`(n)` is embedded in [`Euclidean`](@ref). Similar to the metric and
[`MetricManifold`](@ref), an embedding is often implicitly assumed.

This manifold aims to provide a decorator to model different embeddings of one manifold
and special embeddings of a single manifold. For example an [`IsometricEmbedding`](@ref)
can use the functions from the embedding restricted to the domain.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/PowerManifold.jl"]
Order = [:type, :function]
```
