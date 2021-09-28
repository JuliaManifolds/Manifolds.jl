# Implement Pseudo-Riemannian Metric Symplectic Manifold.

# implement Pseudo-Riemannian metric as subtyupe of AbstracMetric, look at SPD-s.
struct RealSymplecticPseudoRiemannian{n} <: AbstractRealSymplectic{n}
end

# It is possible to make this the default constructor for all structs which subtype from 'AbstractRealSymplectic'? 
RealSymplecticPseudoRiemannian(embedding_dimension) = begin @assert embedding_dimension % 2 == 0; RealSymplecticPseudoRiemannian{div(embedding_dimension, 2)}() end
# check_point(::SymplecticPseudoRiemannian{n}, p; kwargs...) where {n} = check_point(RealSymplectic(n), p; kwargs...)

# check_vector(::SymplecticPseudoRiemannian{n}, p, X; kwargs...) where {n} = check_vector(RealSymplectic(n), p, X; kwargs...)

function exp!(::RealSymplecticPseudoRiemannian, q, p, X)
    p_inv = inv(p)
    q .= p*LinearAlgebra.exp(p_inv * X)
end

