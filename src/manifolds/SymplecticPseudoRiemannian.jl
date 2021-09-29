# Implement Pseudo-Riemannian Metric Symplectic Manifold.

# Implemented Pseudo-Riemannian metric as subtyupe of AbstracMetric, taking inspiration from SPD-manifolds.
struct PseudoRiemannianMetric <: AbstractMetric 
end

function exp!(::MetricManifold{ℝ, <:AbstractSymplectic{n, ℝ}, PseudoRiemannianMetric}, q, p, X) where {n, ℝ}
    println("Called 'exp!()' for PseudoRiemannianMetric.")
    p_inv = inv(p)
    q .= p*LinearAlgebra.exp(p_inv * X)
end
