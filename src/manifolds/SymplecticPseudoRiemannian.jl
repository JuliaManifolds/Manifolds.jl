# Implement Pseudo-Riemannian Metric Symplectic Manifold.

# Implemented Pseudo-Riemannian metric as subtyupe of AbstracMetric, taking inspiration from SPD-manifolds.
struct SymplecticPseudoRiemannianMetric <: AbstractMetric 
end

function exp!(::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, q, p, X) where {n, ℝ}
    p_inv = inv(p)
    q .= p*LinearAlgebra.exp(p_inv * X)
end


@doc """
    inner!(::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, X, Y)

Compute the indefinite inner product giving rise to the pseudoriemannian metric,
known as the Khvedelidze–Mladenov metric. 

# Flesh out.
"""
function inner(::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, X, Y)
    T = Base.promote_eltype(p, X, Y)
    p_inv = inv(p)
    return tr(p_inv * X * p_inv * Y)
end


