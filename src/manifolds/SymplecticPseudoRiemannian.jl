# Implement Pseudo-Riemannian Metric Symplectic Manifold.

# implement Pseudo-Riemannian metric as subtyupe of AbstracMetric, look at SPD-s.
struct SymplecticPseudoRiemannian{n} <: AbstractSymplectic{n, ℝ} 
end

function exp!(::SymplecticPseudoRiemannian, q, p, X)
    p_inv = inv(p)
    q .= p*LinearAlgebra.exp(p_inv * X)
end

