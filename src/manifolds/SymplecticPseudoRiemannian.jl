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
known as the Khvedelidze–Mladenov metric. As referenced in Simone Fiori[^FioriSimone2011].

[^FioriSimone2011]:
    > Simone Fiori:
    > Solving minimal-distance problems over the manifold of real-symplectic matrices,
    > SIAM Journal on Matrix Analysis and Applications 32(3), pp. 938-968, 2011.
    > doi [10.1137/100817115](https://doi.org/10.1137/100817115).
"""
function inner(::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, X, Y) where {n}
    c = cholesky(p)
    return tr((c \ X) * (c \ Y))
end

@doc """
    grad_euclidian_to_manifold(M::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, ∇f_euc)


"""
function grad_euclidian_to_manifold(M::MetricManifold{ℝ, Symplectic{n, ℝ}, SymplecticPseudoRiemannianMetric}, p, ∇f_euc) where {n}
    inner_expression = p' * symplectic_multiply(M.manifold, ∇f_euc; left=false) .- symplectic_multiply(M.manifold, ∇f_euc') * p  
    ∇f_man = (1/2) .* p * symplectic_multiply(M.manifold, inner_expression)
    return ∇f_man
end

