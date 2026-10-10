@doc raw"""
    GeneralizedBurresWassertseinMetric{T<:AbstractMatrix} <: AbstractMetric

The generalized Bures Wasserstein metric for symmetric positive definite matrices, see [HanMishraJawanpuriaGao:2021](@cite).

This metric internally stores the symmetric positive definite matrix ``B`` to generalise the metric,
which is called ``M`` in the cited paper.
"""
struct GeneralizedBuresWassersteinMetric{T <: AbstractMatrix} <: RiemannianMetric
    M::T
    GeneralizedBuresWassersteinMetric(MM::TT) where {TT <: AbstractMatrix} = new{TT}(MM)
end

@doc raw"""
    change_representer(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric}, E::EuclideanMetric, p, X)

Compute the representer of the linear function given by ``X ∈ T_p\mathcal M`` with respect to
the [`GeneralizedBuresWassersteinMetric`](@ref) with matrix ``B`` on the [`SymmetricPositiveDefinite`](@ref) `M`.
Here `X` represents the linear function on the tangent space at `p` with respect to the
[`EuclideanMetric`](@extref `ManifoldsBase.EuclideanMetric`) `g_E`.

To be precise we are looking for ``Z∈T_p\mathcal P(n)`` such that for all ``Y∈T_p\mathcal P(n)``
it holds

```math
⟨X,Y⟩ = \operatorname{tr}(XY) = ⟨Z,Y⟩_{\mathrm{BW}}
```
for all ``Y`` and hence we get ``Z = 2pXB + 2BXp``.
"""
change_representer(
    ::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
    ::EuclideanMetric,
    p,
    X,
)

function change_representer!(
        M::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
        Y,
        ::EuclideanMetric,
        p,
        X,
    )
    Y .= 2 .* (p * X * M.metric.M + M.metric.M * X * p)
    return Y
end

@doc raw"""
    distance(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric}, p, q)

Compute the distance on the [`SymmetricPositiveDefinite`](@ref) manifold `M` with respect to the
[`GeneralizedBuresWassersteinMetric`](@ref) with matrix ``B``, i.e.

```math
d(p,q) = \sqrt{\operatorname{tr}(B^{-1}p) + \operatorname{tr}(B^{-1}q)
       - 2\operatorname{tr}\bigl((B^{-1}qB^{-1}p)^{\frac{1}{2}}\bigr)},
```

see [HuangZheng:2023](@cite), Section 3.2, eqs. (24) and (25).
"""
function distance(
        M::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
        p,
        q,
    )
    luM = lu(M.metric.M)
    luMp = luM \ p
    luMq = luM \ q
    return sqrt(max(real(tr(luMp) + tr(luMq) - 2 * tr(sqrt(luMq * luMp))), 0))
end

@doc raw"""
    exp(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric}, p, X)

Compute the exponential map on the [`SymmetricPositiveDefinite`](@ref) manifold `M` with respect to
the [`GeneralizedBuresWassersteinMetric`](@ref) with matrix ``B``, given in Table 1 of
[HanMishraJawanpuriaGao:2023](@cite) by

```math
    \exp_p(X) = p+X+BL_{p,B}(X)pL_{p,B}(X)B
```

where ``q=L_{p,B}(X)`` denotes the generalized Lyapunov operator, i.e. it solves ``pqB + Bqp = X``,
as defined below Eq. (3) there.
"""
exp(
    ::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
    p,
    X,
)

function exp!(
        M::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
        q,
        p,
        X,
    )
    m = M.metric.M
    Y = lyapc(p, m, -X) #lyap solves qpM + Mpq - X =0
    q .= p .+ X .+ m * Y * p * Y * m
    # symmetrizing for better accuracy
    copyto!(q, (q .+ q') ./ 2)
    return q
end

@doc raw"""
    inner(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric}, p, X, Y)

Compute the inner product on the [`SymmetricPositiveDefinite`](@ref) manifold `M` with respect to
the [`GeneralizedBuresWassersteinMetric`](@ref) with matrix ``B``, given in Eq. (3) of
[HanMishraJawanpuriaGao:2023](@cite) by

```math
    ⟨X,Y⟩ = \frac{1}{2}\operatorname{tr}(L_{p,B}(X)Y)
```

where ``q=L_{p,B}(X)`` denotes the generalized Lyapunov operator, i.e. it solves ``pqB + Bqp = X``,
as defined below Eq. (3) there.
"""
function inner(
        M::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
        p,
        X,
        Y,
    )
    return dot(lyapc(p, M.metric.M, -X), Y) / 2
end

@tfvector_inner_via_get_vector MetricManifold{
    ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric,
}

"""
    is_flat(::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric})

Return false. [`SymmetricPositiveDefinite`](@ref) with [`GeneralizedBuresWassersteinMetric`](@ref)
is not a flat manifold.
"""
function is_flat(
        ::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
    )
    return false
end

@doc raw"""
    log(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,<:GeneralizedBuresWassersteinMetric}, p, q)

Compute the logarithmic map on the [`SymmetricPositiveDefinite`](@ref) manifold `M` with respect to
the [`GeneralizedBuresWassersteinMetric`](@ref) with matrix ``B`` given by

```math
    \log_p(q) = B(B^{-1}pB^{-1}q)^{\frac{1}{2}} + (qB^{-1}pB^{-1})^{\frac{1}{2}}B - 2 p.
```
"""
log(
    ::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
    p,
    q,
)

function log!(
        M::MetricManifold{ℝ, <:SymmetricPositiveDefinite, <:GeneralizedBuresWassersteinMetric},
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
