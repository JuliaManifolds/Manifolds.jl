@doc raw"""
    TODO: Explain this Metric.
"""
struct SymplecticPseudoRiemannianMetric <: AbstractMetric 
end

function exp!(M::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, q, p, X) where {n, ℝ}
    p_inv = inv(M, p)
    q .= p*LinearAlgebra.exp(p_inv * X)
    return q
end


@doc """
    inner!(::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, X, Y)

Compute the indefinite inner product giving rise to the pseudoriemannian metric,
known as the Khvedelidze–Mladenov metric. As referenced in Simone Fiori[^FioriSimone2011].

[^FioriSimone2011]:
    > Simone Fiori:
    > Solving minimal-distance problems over the manifold of real-symplectic matrices,
    > SIAM Journal on Matrix Analysis and Applications 32(3), pp. 938-968, 2011.
    > doi [10.1137/100817115](https://doi.org/10.1137/100817115).
"""
function inner(M::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, X, Y) where {n}
    p_star = symplectic_inverse(M, p)
    return tr((p_star * X) * (p_star * Y))
end


@doc raw"""
    grad_euclidian_to_manifold(M::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, ∇f_euc)

TODO: Flesh out. Inspired by Fiori and Gao et al. Use projection from Gao et al, Metric compatibility from Firoi.
"""
function grad_euclidian_to_manifold(M::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, ∇f_euc) where {n}
    metric_compatible_grad_f = change_representer(M, EuclideanMetric(), p, ∇f_euc)
    return project(M, p, metric_compatible_grad_f)
end

@doc raw"""
    change_representer!(::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, Y, ::EuclideanMetric, p, X)

Change the representer for a tangent vector from the EuclidianMetric to the SymplecticPseudoRiemannianMetric.
TODO: Flesh out.
"""
function change_representer!(::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, Y, ::EuclideanMetric, p, X) where {n}
    Y .= p * ∇f_euc' * p
    return Y
end
