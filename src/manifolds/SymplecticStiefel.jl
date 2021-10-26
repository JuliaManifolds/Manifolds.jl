@doc raw"""

The Symplectic Stiefel manifold. Each element represent a Symplectic Subspace of ``ℝ^{2n × 2k}``. 
"""
struct SymplecticStiefel{n, k, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType} 
end

@doc """
    You are given a manifold of embedding dimension 2n × 2p.
"""
SymplecticStiefel(n::Int, k::Int, field::AbstractNumbers=ℝ) = begin 
    SymplecticStiefel{n, k, field}()
end

decorated_manifold(::SymplecticStiefel{n, k, ℝ}) where {n, k} = Euclidean(2n, 2k; field=ℝ)

function check_point(M::SymplecticStiefel{n, k}, p; kwargs...) where {n, k}
    abstract_embedding_type = supertype(typeof(M))
    
    mpv = invoke(check_point, Tuple{abstract_embedding_type, typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    
    # Perform check that the matrix lives on the real symplectic manifold:
    expected_zero = norm(inv(M, p) * p - I)
    if !isapprox(expected_zero, zero(eltype(p)); kwargs...)
        return DomainError(
            expected_zero,
            ("The point p:\n$(p)\ndoes not lie on $(M) because its symplectic" 
           * " inverse composed with itself is not the identity.")
        )
    end
    return nothing
end

# Document 'check_vector'.
@doc raw"""
    Reference: Proposition 3.2 in Benodkat-Zimmermann. (Eq. 3.3, Second tangent space parametrization.)

````math
    T_p\operatorname{SpSt}(2n, 2n) = \left\{X ∈ ℝ^{2n × 2k} | (p^{+}X)^{+} = -p^{+}X \text{in the Lie Algebra of Hamiltonian Matrices}\right\}
````
"""
check_vector(::SymplecticStiefel, ::Any...)

function check_vector(M::SymplecticStiefel{n, k}, p, X; kwargs...) where {n, k}
    abstract_embedding_type = supertype(typeof(M))

    mpv = invoke(
        check_vector,
        Tuple{abstract_embedding_type, typeof(p), typeof(X)},
        M, p, X;
        kwargs...,
    )
    mpv === nothing || return mpv

    p_star_X = inv(M, p) * X
    hamiltonian_identity_norm = norm(inv(M, p_star_X) + p_star_X)

    if !isapprox(hamiltonian_identity_norm, 0.0; kwargs...)
        return DomainError(
            hamiltonian_identity_norm,
            ("The matrix X: $X is not in the tangent space at point p: $p of the"
           * " $(M) manifold, as p^{+}X is not a Hamiltonian matrix.")
        )
    end
    return nothing
end

Base.inv(::SymplecticStiefel, p) = begin Q = SymplecticMatrix(p); Q' * p' * Q end

