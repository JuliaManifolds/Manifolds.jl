
function active_traits(f, ::ProductManifold, args...)
    return merge_traits(IsDefaultMetric(ProductMetric()))
end

function allocate_coordinates(::ProductManifold, p, T, n::Int)
    return allocate(submanifold_component(p, 1), T, n)
end

function adjoint_Jacobi_field!(M::ProductManifold, Y, p, q, t, X, β::Tβ) where {Tβ}
    map(
        adjoint_Jacobi_field!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, q),
        ntuple(_ -> t, length(M.manifolds)),
        submanifold_components(M, X),
        ntuple(_ -> β, length(M.manifolds)),
    )
    return Y
end

@doc raw"""
    flat(M::ProductManifold, p, X::FVector{TangentSpaceType})

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on the [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`)
`M` to a cotangent vector. This can be done elementwise for every entry of `X` (with respect
to the corresponding entry in `p`) separately.
"""
flat(::ProductManifold, ::Any...)

function jacobi_field!(M::ProductManifold, Y, p, q, t, X, β::Tβ) where {Tβ}
    map(
        jacobi_field!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, q),
        ntuple(_ -> t, length(M.manifolds)),
        submanifold_components(M, X),
        ntuple(_ -> β, length(M.manifolds)),
    )
    return Y
end

"""
    manifold_volume(M::ProductManifold)

Return the volume of [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`)
`M`, i.e. product of volumes of the manifolds `M` is constructed from.
"""
manifold_volume(M::ProductManifold) = mapreduce(manifold_volume, *, M.manifolds)

function riemannian_gradient!(M::ProductManifold, Y, p, X)
    map(
        riemannian_gradient!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return Y
end

@doc raw"""
    Y = riemannian_Hessian(M::ProductManifold, p, G, H, X)
    riemannian_Hessian!(M::ProductManifold, Y, p, G, H, X)

Compute the Riemannian Hessian ``\operatorname{Hess} f(p)[X]`` given the
Euclidean gradient ``∇ f(\tilde p)`` in `G` and the Euclidean Hessian ``∇^2 f(\tilde p)[\tilde X]`` in `H`,
where ``\tilde p, \tilde X`` are the representations of ``p,X`` in the embedding,.

On a product manifold, this decouples and can be computed elementwise.
"""
riemannian_Hessian(M::ProductManifold, p, G, H, X)

function riemannian_Hessian!(M::ProductManifold, Y, p, G, H, X)
    map(
        riemannian_Hessian!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, G),
        submanifold_components(M, H),
        submanifold_components(M, X),
    )
    return Y
end

@doc raw"""
    sharp(M::ProductManifold, p, ξ::FVector{CotangentSpaceType})

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on the [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`)
`M` to a tangent vector. This can be done elementwise for every entry of `ξ` (and `p`)
separately
"""
sharp(::ProductManifold, ::Any...)

@doc raw"""
    volume_density(M::ProductManifold, p, X)

Return volume density on the [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`)
`M`, i.e. product of constituent volume densities.
"""
function volume_density(M::ProductManifold, p, X)
    dens = map(
        volume_density,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return prod(dens)
end
