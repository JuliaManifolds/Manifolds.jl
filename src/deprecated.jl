@doc raw"""
    det_local_metric(M::AbstractManifold, p, B::AbstractBasis)

Return the determinant of local matrix representation of the metric tensor ``g``, i.e. of the
matrix ``G(p)`` representing the metric in the tangent space at ``p`` with as a matrix.

See also [`local_metric`](@ref)
"""
function det_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    @warn "`det_local_metric(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `det_local_metric(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
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
    @warn "`einstein_tensor(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `einstein_tensor(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
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
    @warn "`inverse_local_metric(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `inverse_local_metric(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
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
    @warn "`local_metric(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `local_metric(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
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
local_metric_jacobian(::AbstractManifold, ::Any, B::AbstractBasis)
function local_metric_jacobian(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    @warn "`local_metric_jacobian(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `christoffel_symbols_second(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
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
    @warn "`log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `log_local_metric_density(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
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
    @warn "`ricci_curvature(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `ricci_curvature(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
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
for mf in [
        christoffel_symbols_second_jacobian,
        local_metric_jacobian,
    ]
    @eval is_metric_function(::typeof($mf)) = true
end
