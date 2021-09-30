# Implement Pseudo-Riemannian Metric Symplectic Manifold.

# Implemented Pseudo-Riemannian metric as subtyupe of AbstracMetric, taking inspiration from SPD-manifolds.
struct SymplecticPseudoRiemannianMetric <: AbstractMetric 
end

function exp!(::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, q, p, X) where {n, ℝ}
    p_inv = inv(p)
    q .= p*LinearAlgebra.exp(p_inv * X)
end
