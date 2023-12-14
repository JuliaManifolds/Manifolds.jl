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

function ProductFVectorDistribution(distributions::FVectorDistribution...)
    M = ProductManifold(map(d -> support(d).space.manifold, distributions)...)
    fiber_type = support(distributions[1]).space.fiber_type
    if !all(d -> support(d).space.fiber_type == fiber_type, distributions)
        error(
            "Not all distributions have support in vector spaces of the same type, which is currently not supported",
        )
    end
    # Probably worth considering sum spaces in the future?
    p = ArrayPartition(map(d -> support(d).space.point, distributions)...)
    return ProductFVectorDistribution(Fiber(M, p, fiber_type), distributions)
end

function rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ArrayPartition(map(d -> rand(rng, d), d.distributions)...)
end
function rand(rng::AbstractRNG, d::ProductFVectorDistribution)
    return ArrayPartition(map(d -> rand(rng, d), d.distributions)...)
end
function _rand!(rng::AbstractRNG, d::ProductPointDistribution, p::ArrayPartition)
    map(
        (t1, t2) -> Distributions._rand!(rng, t1, t2),
        d.distributions,
        submanifold_components(d.manifold, p),
    )
    return p
end
function _rand!(rng::AbstractRNG, d::ProductFVectorDistribution, X::ArrayPartition)
    map(
        t -> Distributions._rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.space.manifold, X),
    )
    return X
end
