function change_metric!(M::AbstractManifold, Y, G::AbstractMetric, p, X)
    metric(M) === G && return copyto!(M, Y, p, X) # no metric change
    # TODO: For local metric, inverse_local metric, det_local_metric: Introduce a default basis?
    B = DefaultOrthogonalBasis()
    G1 = local_metric(M, p, B)
    G2 = local_metric(G(M), p, B)
    x = get_coordinates(M, p, X, B)
    C1 = cholesky(G1).L
    C2 = cholesky(G2).L
    z = (C1 \ C2)'x
    return get_vector!(M, Y, p, z, B)
end

function change_representer!(M::AbstractManifold, Y, G::AbstractMetric, p, X)
    (metric(M) == G) && return copyto!(M, Y, p, X) # no metric change
    # TODO: For local metric, inverse_local metric, det_local_metric: Introduce a default basis?
    B = DefaultOrthogonalBasis()
    G1 = local_metric(M, p, B)
    G2 = local_metric(G(M), p, B)
    x = get_coordinates(M, p, X, B)
    z = (G1 \ G2)'x
    return get_vector!(M, Y, p, z, B)
end

@doc raw"""
    flat(N::MetricManifold{M,G}, p, X::TFVector)

Compute the musical isomorphism to transform the tangent vector `X` from the
[`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` equipped with
[`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) `G` to a cotangent by
computing

````math
X^♭= G_p X,
````
where ``G_p`` is the local matrix representation of `G`, see [`local_metric`](@ref)
"""
flat(::MetricManifold, ::Any, ::TFVector)

function flat!(M::AbstractManifold, ξ::CoTFVector, p, X::TFVector)
    (metric(M.manifold) == M.metric) && (return flat!(M.manifold, ξ, p, X))
    g = local_metric(M, p, ξ.basis)
    copyto!(ξ.data, g * X.data)
    return ξ
end

function inner(M::MetricManifold, p, X::TFVector, Y::TFVector)
    X.basis === Y.basis ||
        error("calculating inner product of vectors from different bases is not supported")
    return dot(X.data, local_metric(M, p, X.basis) * Y.data)
end

@doc raw"""
    sharp(N::MetricManifold{M,G}, p, ξ::CoTFVector)

Compute the musical isomorphism to transform the cotangent vector `ξ` from the
[`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` equipped with
[`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) `G` to a tangent by
computing

````math
ξ^♯ = G_p^{-1} ξ,
````
where ``G_p`` is the local matrix representation of `G`, i.e. one employs
[`inverse_local_metric`](@ref) here to obtain ``G_p^{-1}``.
"""
sharp(::MetricManifold, ::Any, ::CoTFVector)

function sharp!(M::MetricManifold, X::TFVector, p, ξ::CoTFVector)
    (metric(M.manifold) == M.metric) && (return sharp!(M.manifold, X, p, ξ))
    Ginv = inverse_local_metric(M, p, X.basis)
    copyto!(X.data, Ginv * ξ.data)
    return X
end
