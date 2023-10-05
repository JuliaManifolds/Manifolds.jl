
function active_traits(f, ::ProductManifold, args...)
    return merge_traits(IsDefaultMetric(ProductMetric()))
end

function allocate_coordinates(::ProductManifold, p, T, n::Int)
    return allocate(submanifold_component(p, 1), T, n)
end

"""
    ProductFVectorDistribution([type::VectorBundleFibers], [x], distrs...)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the product distribution of given distributions.

Vector space type and `x` can be automatically inferred from distributions `distrs`.
"""
struct ProductFVectorDistribution{
    TSpace<:VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    TD<:(NTuple{N,Distribution} where {N}),
    TX,
} <: FVectorDistribution{TSpace,TX}
    type::TSpace
    x::TX
    distributions::TD
end

"""
    ProductPointDistribution(M::ProductManifold, distributions)

Product distribution on manifold `M`, combined from `distributions`.
"""
struct ProductPointDistribution{
    TM<:ProductManifold,
    TD<:(NTuple{N,Distribution} where {N}),
} <: MPointDistribution{TM}
    manifold::TM
    distributions::TD
end

function adjoint_Jacobi_field(
    M::ProductManifold,
    p::ArrayPartition,
    q::ArrayPartition,
    t,
    X::ArrayPartition,
    β::Tβ,
) where {Tβ}
    return ArrayPartition(
        map(
            adjoint_Jacobi_field,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            ntuple(_ -> t, length(M.manifolds)),
            submanifold_components(M, X),
            ntuple(_ -> β, length(M.manifolds)),
        )...,
    )
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
    cross(M, N)
    cross(M1, M2, M3,...)

Return the [`ProductManifold`](@ref) For two `AbstractManifold`s `M` and `N`,
where for the case that one of them is a [`ProductManifold`](@ref) itself,
the other is either prepended (if `N` is a product) or appenden (if `M`) is.
If both are product manifold, they are combined into one product manifold,
keeping the order.

For the case that more than one is a product manifold of these is build with the
same approach as above
"""
cross(::AbstractManifold...)
LinearAlgebra.cross(M1::AbstractManifold, M2::AbstractManifold) = ProductManifold(M1, M2)
function LinearAlgebra.cross(M1::ProductManifold, M2::AbstractManifold)
    return ProductManifold(M1.manifolds..., M2)
end
function LinearAlgebra.cross(M1::AbstractManifold, M2::ProductManifold)
    return ProductManifold(M1, M2.manifolds...)
end
function LinearAlgebra.cross(M1::ProductManifold, M2::ProductManifold)
    return ProductManifold(M1.manifolds..., M2.manifolds...)
end

@doc raw"""
    flat(M::ProductManifold, p, X::FVector{TangentSpaceType})

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on the [`ProductManifold`](@ref) `M` to a cotangent vector.
This can be done elementwise for every entry of `X` (with respect to the corresponding
entry in `p`) separately.
"""
flat(::ProductManifold, ::Any...)

function jacobi_field(
    M::ProductManifold,
    p::ArrayPartition,
    q::ArrayPartition,
    t,
    X::ArrayPartition,
    β::Tβ,
) where {Tβ}
    return ArrayPartition(
        map(
            jacobi_field,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
            ntuple(_ -> t, length(M.manifolds)),
            submanifold_components(M, X),
            ntuple(_ -> β, length(M.manifolds)),
        )...,
    )
end
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

Return the volume of [`ProductManifold`](@ref) `M`, i.e. product of volumes of the
manifolds `M` is constructed from.
"""
manifold_volume(M::ProductManifold) = mapreduce(manifold_volume, *, M.manifolds)

function ProductFVectorDistribution(
    type::VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    p::Union{AbstractArray,AbstractManifoldPoint},
    distributions::FVectorDistribution...,
)
    return ProductFVectorDistribution{typeof(type),typeof(distributions),typeof(p)}(
        type,
        p,
        distributions,
    )
end
function ProductFVectorDistribution(
    type::VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    distributions::FVectorDistribution...,
)
    p = ArrayPartition(map(d -> support(d).point, distributions))
    return ProductFVectorDistribution(type, p, distributions...)
end
function ProductFVectorDistribution(distributions::FVectorDistribution...)
    M = ProductManifold(map(d -> support(d).space.manifold, distributions)...)
    fiber = support(distributions[1]).space.fiber
    if !all(d -> support(d).space.fiber == fiber, distributions)
        error(
            "Not all distributions have support in vector spaces of the same type, which is currently not supported",
        )
    end
    # Probably worth considering sum spaces in the future?
    x = ArrayPartition(map(d -> support(d).point, distributions)...)
    return ProductFVectorDistribution(VectorBundleFibers(fiber, M), x, distributions...)
end

function ProductPointDistribution(M::ProductManifold, distributions::MPointDistribution...)
    return ProductPointDistribution{typeof(M),typeof(distributions)}(M, distributions)
end
function ProductPointDistribution(distributions::MPointDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    return ProductPointDistribution(M, distributions...)
end

function Random.rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ArrayPartition(map(d -> rand(rng, d), d.distributions)...)
end
function Random.rand(rng::AbstractRNG, d::ProductFVectorDistribution)
    return ArrayPartition(map(d -> rand(rng, d), d.distributions)...)
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductPointDistribution,
    x::AbstractArray{<:Number},
)
    return copyto!(x, rand(rng, d))
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductPointDistribution,
    p::ArrayPartition,
)
    map(
        t -> Distributions._rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.manifold, p),
    )
    return p
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductFVectorDistribution,
    v::AbstractArray{<:Number},
)
    return copyto!(v, rand(rng, d))
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductFVectorDistribution,
    X::ArrayPartition,
)
    map(
        t -> Distributions._rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.space.manifold, X),
    )
    return X
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
`p` on the [`ProductManifold`](@ref) `M` to a tangent vector.
This can be done elementwise for every entry of `ξ` (and `p`) separately
"""
sharp(::ProductManifold, ::Any...)

Distributions.support(d::ProductPointDistribution) = MPointSupport(d.manifold)
function Distributions.support(tvd::ProductFVectorDistribution)
    return FVectorSupport(
        tvd.type,
        ArrayPartition(map(d -> support(d).point, tvd.distributions)...),
    )
end

function uniform_distribution(M::ProductManifold)
    return ProductPointDistribution(M, map(uniform_distribution, M.manifolds))
end
function uniform_distribution(M::ProductManifold, p)
    return ProductPointDistribution(
        M,
        map(uniform_distribution, M.manifolds, submanifold_components(M, p)),
    )
end

@doc raw"""
    volume_density(M::ProductManifold, p, X)

Return volume density on the [`ProductManifold`](@ref) `M`, i.e. product of constituent
volume densities.
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
