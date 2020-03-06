# Embedded manifold

Some manifolds can easily be defined by using a certain embedding.
For example the [`Sphere`](@ref)`(n)` is embedded in [`Euclidean`](@ref)`(n+1)`.
Similar to the metric and [`MetricManifold`](@ref), an embedding is often implicitly assumed.
We introduce the embedded manifolds hence as an [`AbstractDecoratorManifold`](@ref).

This decorator enables to use such an embedding in an transparent way.
Different types of embeddings can be distinguished using the [`AbstractEmbeddingType`](@ref).

The embedding also covers representation of tangent vectors.
For both points and tangent vectors the function [`embed`](@ref) returns their representation in the embedding.
For any point or vector in the embedding the functions [`project_point`](@ref) and [`project_tangent`](@ref) can be used to obtain the closest point on the manifold and tangent vector in the tangent space, respectively.
A specific example where [`embed`](@ref) might be useful, is for example a Lie group, where tangent vectors are often represented in the Lie algebra.
Then their representation is different from the representation in the embedding.

## Isometric Embeddings

For isometric embeddings the type [`AbstractIsometricEmbeddingType`](@ref) can be used to avoid reimplementing the metric.
See [`Sphere`](@ref) or [`Hyperbolic`](@ref) for example.
Here, the exponential map, the logarithmic map, the retraction and its inverse
are set to `:intransparent`, i.e. they have to be implemented.

Furthermore, the [`TransparentIsometricEmbedding`](@ref) type even states that the exponential
and logarithmic maps as well as retractions and vector transports of the embedding can be
used for the embedded manifold as well.
See [`SymmetricMatrices`](@ref) for an example.

In both cases of course [`check_manifold_point`](@ref) and [`check_tangent_vector`](@ref) have to be implemented.

## Technical Details

Semantically we use the idea of the embedding to efficiently implement a manifold by not having to implement those functions that are already given by its embedding. Hence we decorate in some sense the manifold we implement.
Still, technicall [`base_manifold`](@ref) is the embedding as long as [`EmbeddedManifold`](@ref) is used.
For the abstract case, [`AbstractEmbeddedManifold`](@ref) the base manfiold might differ.
Note that internally [`base_manifold`](@ref) uses [`decorated_manifold`](@ref) for one step of removing multiple decorators.

Clearly [`get_embedding`](@ref) always returns the embedding.

## Types

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/EmbeddedManifold.jl"]
Order = [:type]
```

## Functions

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/EmbeddedManifold.jl"]
Order = [:function]
```
