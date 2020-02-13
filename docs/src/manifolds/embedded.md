# Embedded manifold

The A lot of manifolds can easily be defined in their embedding. For example the
[`Sphere`](@ref)`(n)` is embedded in [`Euclidean`](@ref). Similar to the metric and
[`MetricManifold`](@ref), an embedding is often implicitly assumed.

This decorator enables to use such an embedding in an transparent way. A manifold can be defined using the embedding, see [`SymmetricMatrices`](@ref), where just the functions that are different from the embedding have to be implemented.

This also covers representation of tangent vectors. For these transforms [`embed`](@ref) and [`project_point`](@ref) and [`project_tangent`](@ref) can be used. The last two might often already be implemented. [`embed`](@ref) might be useful, when for example a Lie group tangent vector is represented within the Lie algebra and in the embedding this has to be parallel transported from the [`Identity`](@ref) to the point the tangent space is attached to.

Further, different embeddings can be modeled, especially to change representation of points between a manifold and its embedding.
For example an [`IsometricEmbedding`](@ref)
can use the functions from the embedding restricted to the domain.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Embedded.jl"]
Order = [:type, :function]
```
