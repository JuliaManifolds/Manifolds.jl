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
    det_local_metric(M::AbstractManifold, p, B::AbstractBasis)

Return the determinant of local matrix representation of the metric tensor ``g``, i.e. of the
matrix ``G(p)`` representing the metric in the tangent space at ``p`` with as a matrix.

See also [`local_metric`](@ref)
"""
function det_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    return det(local_metric(M, p, B))
end

"""
    einstein_tensor(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = diff_badefault_differential_backendckend())

Compute the Einstein tensor of the manifold `M` at the point `p`, see [https://en.wikipedia.org/wiki/Einstein_tensor](https://en.wikipedia.org/wiki/Einstein_tensor)
"""
function einstein_tensor(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    Ric = ricci_tensor(M, p, B; backend = backend)
    g = local_metric(M, p, B)
    Ginv = inverse_local_metric(M, p, B)
    S = sum(Ginv .* Ric)
    G = Ric - g .* S / 2
    return G
end
@trait_function einstein_tensor(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

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
    inverse_local_metric(M::AbstractManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation of the inverse metric (cometric) tensor
of the tangent space at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` with respect
to the [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) basis `B`.

The metric tensor (see [`local_metric`](@ref)) is usually denoted by ``G = (g_{ij}) ∈ 𝔽^{d×d}``,
where ``d`` is the dimension of the manifold.

Then the inverse local metric is denoted by ``G^{-1} = g^{ij}``.
"""
inverse_local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
function inverse_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    return inv(local_metric(M, p, B))
end
@trait_function inverse_local_metric(M::AbstractDecoratorManifold, p, B::AbstractBasis)


@doc raw"""
    local_metric(M::AbstractManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation at the point `p` of the metric tensor ``g`` with
respect to the [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `B` on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M`.
Let ``d``denote the dimension of the manifold and $b_1,\ldots,b_d$ the basis vectors.
Then the local matrix representation is a matrix ``G\in 𝔽^{n×n}`` whose entries are
given by ``g_{ij} = g_p(b_i,b_j), i,j\in\{1,…,d\}``.

This yields the property for two tangent vectors (using Einstein summation convention)
``X = X^ib_i, Y=Y^ib_i \in T_p\mathcal M`` we get ``g_p(X, Y) = g_{ij} X^i Y^j``.
"""
local_metric(::AbstractManifold, ::Any, ::AbstractBasis)

function local_metric(M::MetricManifold, p, B::AbstractBasis)
    (metric(M.manifold) == M.metric) && (return local_metric(M.manifold, p, B))
    return invoke(local_metric, Tuple{AbstractManifold, Any, AbstractBasis}, M, p, B)
end

@doc raw"""
    local_metric_jacobian(M::AbstractManifold, p, B::AbstractBasis;
    backend::AbstractDiffBackend,
)

Get partial derivatives of the local metric of `M` at `p` in basis `B` with respect to the
coordinates of `p`, ``\frac{∂}{∂ p^k} g_{ij} = g_{ij,k}``. The
dimensions of the resulting multi-dimensional array are ordered ``(i,j,k)``.
"""
local_metric_jacobian(::AbstractManifold, ::Any, B::AbstractBasis, ::AbstractDiffBackend)
function local_metric_jacobian(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    n = size(p, 1)
    ∂g = reshape(_jacobian(q -> local_metric(M, q, B), copy(M, p), backend), n, n, n)
    return ∂g
end

@doc raw"""
    log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)

Return the natural logarithm of the metric density ``ρ`` of `M` at `p`, which
is given by ``ρ = \log \sqrt{|\det [g_{ij}]|}`` for the metric tensor expressed in basis `B`.
"""
log_local_metric_density(::AbstractManifold, ::Any, ::AbstractBasis)
function log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)
    return log(abs(det_local_metric(M, p, B))) / 2
end

function norm(M::MetricManifold, p, X::TFVector)
    return sqrt(dot(X.data, local_metric(M, p, X.basis) * X.data))
end

@doc raw"""
    ricci_curvature(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = default_differential_backend())

Compute the Ricci scalar curvature of the manifold `M` at the point `p` using basis `B`.
The curvature is computed as the trace of the Ricci curvature tensor with respect to
the metric, that is ``R=g^{ij}R_{ij}`` where ``R`` is the scalar Ricci curvature at `p`,
``g^{ij}`` is the inverse local metric (see [`inverse_local_metric`](@ref)) at `p` and
``R_{ij}`` is the Ricci curvature tensor, see [`ricci_tensor`](@ref). Both the tensor and
inverse local metric are expressed in local coordinates defined by `B`, and the formula
uses the Einstein summation convention.
"""
ricci_curvature(::AbstractManifold, ::Any, ::AbstractBasis)
function ricci_curvature(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    Ginv = inverse_local_metric(M, p, B)
    Ric = ricci_tensor(M, p, B; backend = backend)
    S = sum(Ginv .* Ric)
    return S
end
ManifoldsBase.@trait_function ricci_curvature(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

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

for mf in [
        christoffel_symbols_first,
        christoffel_symbols_second,
        christoffel_symbols_second_jacobian,
        det_local_metric,
        einstein_tensor,
        flat!,
        gaussian_curvature,
        inverse_local_metric,
        local_metric,
        local_metric_jacobian,
        log_local_metric_density,
        mean,
        mean!,
        median,
        median!,
        ricci_curvature,
        ricci_tensor,
        riemann_tensor,
        riemannian_gradient,
        riemannian_gradient!,
        riemannian_Hessian,
        riemannian_Hessian!,
        sharp!,
    ]
    @eval is_metric_function(::typeof($mf)) = true
end
