@doc raw"""
    GeneralizedBurresWassertseinMetric{T<:AbstractMatrix} <: AbstractMetric

The generalized Bures Wasserstein metric for symmetric positive definite matrices, see[^HanMishraJawanpuriaGao2021].

This metric internally stores the symmetric positive definite matrix ``M`` to generalise the metric,
where the name also follows the mentioned preprint.

[^HanMishraJawanpuriaGao2021]:
    > Han, A., Mishra, B., Jawanpuria, P., Gao, J.:
    > _Generalized Bures-Wasserstein geometry for positive definite matrices_.
    > arXiv: [2110.10464](https://arxiv.org/abs/2110.10464).
"""
struct GeneralizedBuresWassersteinMetric{T<:AbstractMatrix} <: RiemannianMetric
    M::T
    GeneralizedBuresWassersteinMetric(MM::TT) where {TT<:AbstractMatrix} = new{TT}(MM)
end

@doc raw"""
    change_representer(M::MetricManifold{ℝ,SymmetricPositiveDefinite,GeneralizedBuresWassersteinMetric}, E::EuclideanMetric, p, X)

Given a tangent vector ``X ∈ T_p\mathcal M`` representing a linear function on the tangent
space at `p` with respect to the [`EuclideanMetric`](@ref) `g_E`,
this is turned into the representer with respect to the (default) metric,
the [`GeneralizedBuresWassersteinMetric`](@ref) on the [`SymmetricPositiveDefinite`](@ref) `M`.

To be precise we are looking for ``Z∈T_p\mathcal P(n)`` such that for all ``Y∈T_p\mathcal P(n)``
it holds

```math
⟨X,Y⟩ = \operatorname{tr}(XY) = ⟨Z,Y⟩_{\mathrm{BW}}
```
for all ``Y`` and hence we get ``Z = 2pXM + 2MXp``.
"""
change_representer(
    ::MetricManifold{ℝ,SymmetricPositiveDefinite,GeneralizedBuresWassersteinMetric},
    ::EuclideanMetric,
    p,
    X,
)

function change_representer!(
    M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric},
    Y,
    ::EuclideanMetric,
    p,
    X,
)
    Y .= 2 .* (p * X * M.metric.M + M.metric.M * X * p)
    return Y
end

@doc raw"""
    distance(::MatricManifold{SymmetricPositiveDefinite,GeneralizedBuresWassersteinMetric}, p, q)

Compute the distance with respect to the [`BuresWassersteinMetric`](@ref) on [`SymmetricPositiveDefinite`](@ref) matrices, i.e.

```math
d(p,q) = \operatorname{tr}(M^{-1}p) + \operatorname{tr}(M^{-1}q)
       - 2\operatorname{tr}\bigl( (p^{\frac{1}{2}}M^{-1}qM^{-1}p^{\frac{1}{2}} \bigr)^{\frac{1}{2}},
```
"""
function distance(
    M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric},
    p,
    q,
)
    luM = lu(M.metric.M)
    luMp = luM \ p
    luMq = luM \ q
    return sqrt(tr(luMp) + tr(luMq) - 2 * tr(sqrt(luMq * luMp)))
end

@doc raw"""
    exp(::MatricManifold{ℝ,SymmetricPositiveDefinite,GeneralizedBuresWassersteinMetric}, p, X)

Compute the exponential map on [`SymmetricPositiveDefinite`](@ref) with respect to
the [`GeneralizedBuresWassersteinMetric`](@ref) given by

```math
    \exp_p(X) = p+X+\mathcal ML_{p,M}(X)pML_{p,M}(X)
```

where ``q=L_{M,p}(X)`` denotes the generalized Lyapunov operator, i.e. it solves ``pqM + Mqp = X``.
"""
exp(::MetricManifold{ℝ,SymmetricPositiveDefinite,GeneralizedBuresWassersteinMetric}, p, X)

function exp!(
    M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric},
    q,
    p,
    X,
)
    m = M.metric.M
    Y = lyapc(p, m, -X) #lyap solves qpM + Mpq - X =0
    q .= p .+ X .+ m * Y * p * Y * m
    return q
end

@doc raw"""
    inner(::MetricManifold{ℝ,SymmetricPositiveDefinite,GeneralizedBuresWassersteinMetric}, p, X, Y)

Compute the inner product [`SymmetricPositiveDefinite`](@ref) with respect to
the [`GeneralizedBuresWassersteinMetric`](@ref) given by

```math
    ⟨X,Y⟩ = \frac{1}{2}\operatorname{tr}(L_{p,M}(X)Y)
```

where ``q=L_{M,p}(X)`` denotes the generalized Lyapunov operator, i.e. it solves ``pqM + Mqp = X``.
"""
function inner(
    M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric},
    p,
    X,
    Y,
)
    return dot(lyapc(p, M.metric.M, -X), Y) / 2
end

@doc raw"""
    log(::MatricManifold{SymmetricPositiveDefinite,GeneralizedBuresWassersteinMetric}, p, q)

Compute the logarithmic map on [`SymmetricPositiveDefinite`](@ref) with respect to
the [`BuresWassersteinMetric`](@ref) given by

```math
    \log_p(q) = M(M^{-1}pM^{-1}q)^{\frac{1}{2}} + (qM^{-1}pM^{-1})^{\frac{1}{2}}M - 2 p.
```
"""
log(::MetricManifold{ℝ,SymmetricPositiveDefinite,GeneralizedBuresWassersteinMetric}, p, q)

function log!(
    M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric},
    X,
    p,
    q,
)
    m = M.metric.M
    lum = lu(m)
    lum_p_lum = lum \ p / lum
    X .= real.(Symmetric(m * sqrt(lum_p_lum * q) + sqrt(q * lum_p_lum) * m) - 2 * p)
    return X
end
