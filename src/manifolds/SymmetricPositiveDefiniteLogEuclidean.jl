@doc raw"""
    LogEuclideanMetric <: RiemannianMetric

The LogEuclidean Metric consists of the Euclidean metric applied to all elements after mapping them
into the Lie Algebra, i.e. performing a matrix logarithm beforehand.
"""
struct LogEuclideanMetric <: RiemannianMetric end

@doc raw"""
    distance(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,LogEuclideanMetric}, p, q)

Compute the distance on the [`SymmetricPositiveDefinite`](@ref) manifold between
`p` and `q` as a [`MetricManifold`](@extref ManifoldsBase.MetricManifold) with [`LogEuclideanMetric`](@ref).
The formula reads

```math
    d_{\mathcal P(n)}(p,q) = \lVert \operatorname{Log} p - \operatorname{Log} q \rVert_{\mathrm{F}}
```

where $\operatorname{Log}$ denotes the matrix logarithm and
$\lVert⋅\rVert_{\mathrm{F}}$ denotes the matrix Frobenius norm.
"""
function distance(::MetricManifold{ℝ, <:SymmetricPositiveDefinite, LogEuclideanMetric}, p, q)
    return norm(log(Symmetric(p)) - log(Symmetric(q)))
end

"""
    is_flat(::MetricManifold{ℝ,<:SymmetricPositiveDefinite,LogEuclideanMetric})

Return true. [`SymmetricPositiveDefinite`](@ref) with [`LogEuclideanMetric`](@ref)
is a flat manifold: its metric is the pullback of an inner product on the symmetric matrices
by the matrix logarithm, so it is isometric to a Euclidean space and its curvature is zero,
see Section 3.2 and Table 4 of [ThanwerdasPennec:2023](@cite).
"""
is_flat(M::MetricManifold{ℝ, <:SymmetricPositiveDefinite, LogEuclideanMetric}) = true

@doc raw"""
    riemann_tensor(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,LogEuclideanMetric}, p, X, Y, Z)

Compute the Riemann tensor ``R(X,Y)Z`` at `p` on the [`SymmetricPositiveDefinite`](@ref)
matrices `M` with the [`LogEuclideanMetric`](@ref). Its value is always the zero tangent
vector, since ``R = 0`` for this metric, see Table 4 of [ThanwerdasPennec:2023](@cite).
"""
riemann_tensor(
    M::MetricManifold{ℝ, <:SymmetricPositiveDefinite, LogEuclideanMetric}, p, X, Y, Z,
)

function riemann_tensor!(
        ::MetricManifold{ℝ, <:SymmetricPositiveDefinite, LogEuclideanMetric},
        Xresult,
        p,
        X,
        Y,
        Z,
    )
    return fill!(Xresult, 0)
end

@doc raw"""
    sectional_curvature(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,LogEuclideanMetric}, p, X, Y)

Return the sectional curvature ``κ_p(X,Y) = 0`` of the [`SymmetricPositiveDefinite`](@ref) matrices `M` with the [`LogEuclideanMetric`](@ref).
The Riemann tensor of this metric is ``R = 0``, see Table 4 of [ThanwerdasPennec:2023](@cite),
so ``⟨R(X,Y)Y,X⟩_p = 0`` for all linearly independent ``X, Y ∈ T_p\mathcal M``.
"""
function sectional_curvature(
        ::MetricManifold{ℝ, <:SymmetricPositiveDefinite, LogEuclideanMetric}, p, X, Y,
    )
    return 0.0
end

@doc raw"""
    sectional_curvature_max(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,LogEuclideanMetric})

Return the upper bound 0 of the sectional curvature of the [`SymmetricPositiveDefinite`](@ref) matrices `M` with the [`LogEuclideanMetric`](@ref).
The Riemann tensor of this metric is ``R = 0``, see Table 4 of [ThanwerdasPennec:2023](@cite),
so every sectional curvature is 0.
"""
function sectional_curvature_max(
        ::MetricManifold{ℝ, <:SymmetricPositiveDefinite, LogEuclideanMetric},
    )
    return 0.0
end

@doc raw"""
    sectional_curvature_min(M::MetricManifold{ℝ,<:SymmetricPositiveDefinite,LogEuclideanMetric})

Return the lower bound 0 of the sectional curvature of the [`SymmetricPositiveDefinite`](@ref) matrices `M` with the [`LogEuclideanMetric`](@ref).
The Riemann tensor of this metric is ``R = 0``, see Table 4 of [ThanwerdasPennec:2023](@cite),
so every sectional curvature is 0.
"""
function sectional_curvature_min(
        ::MetricManifold{ℝ, <:SymmetricPositiveDefinite, LogEuclideanMetric},
    )
    return 0.0
end
