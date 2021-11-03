@doc raw"""

The Symplectic Stiefel manifold. Each element represent a Symplectic Subspace of ``ℝ^{2n × 2k}``. 
"""
struct SymplecticStiefel{n, k, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType} 
end

#TODO: Implement exponential mapping from Bendokat-Zimmermann.

@doc """
    You are given a manifold of embedding dimension 2n × 2p.
"""
SymplecticStiefel(n::Int, k::Int, field::AbstractNumbers=ℝ) = begin 
    SymplecticStiefel{n, k, field}()
end
Base.show(io::IO, ::SymplecticStiefel{n, k}) where {n, k} = print(io, "SymplecticStiefel{$(2n), $(2k)}()")

decorated_manifold(::SymplecticStiefel{n, k, ℝ}) where {n, k} = Euclidean(2n, 2k; field=ℝ)

@doc raw"""
    manifold_dimension(::SymplecticStiefel{n, k})    

As shown in proposition 3.1 in Bendokat-Zimmermann.
"""
manifold_dimension(::SymplecticStiefel{n, k}) where {n, k} = (4n -2k + 1)*k 

function check_point(M::SymplecticStiefel{n, k}, p; kwargs...) where {n, k}
    abstract_embedding_type = supertype(typeof(M))
    
    mpv = invoke(check_point, Tuple{abstract_embedding_type, typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    
    # Perform check that the matrix lives on the real symplectic manifold:
    expected_zero = norm(inv(M, p) * p - I)
    if !isapprox(expected_zero, 0; kwargs...)
        return DomainError(
            expected_zero,
            ("The point p does not lie on $(M) because its symplectic" 
           * " inverse composed with itself is not the identity. $(p)")
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

    if !isapprox(hamiltonian_identity_norm, 0; kwargs...)
        return DomainError(
            hamiltonian_identity_norm,
            ("The matrix X: $X is not in the tangent space at point p: $p of the"
           * " $(M) manifold, as p^{+}X is not a Hamiltonian matrix.")
        )
    end
    return nothing
end

@doc raw"""
    inner(M::SymplecticStiefel{n, k}, p, X. Y)

Based on the inner product in Proposition 3.10 of Benodkat-Zimmermann.
"""
function inner_3(::SymplecticStiefel{n, k}, p, X, Y) where {n, k}
    Q = SymplecticMatrix(p, X, Y)
    I = UniformScaling(2n)

    # Saves time compared to recomputing (p' * p) and solving a linear system twice:
    inv_pT_p = inv(p' * p) # 𝞞((2k)^3)?

    return tr(X' * (I - (1/2) * Q' * p * inv_pT_p * p' * Q) * Y * inv_pT_p) 
end

function inner_2(::SymplecticStiefel{n, k}, p, X, Y) where {n, k}
    Q = SymplecticMatrix(p, X, Y)
    I = UniformScaling(2n)
    # p_Tp = lu(p' * p)
    # inner_matrix = (I - (1/2) * Q' * p * (p_Tp \ (p')) * Q) 
    # return tr(X' * (I - (1/2) * Q' * p * (p_Tp \ (p')) * Q) * (Y / p_Tp))
    return tr(X' * (I - (1/2) * Q' * p * ((p' * p) \ (p')) * Q) * (Y / (p' * p)))
end

function inner(::SymplecticStiefel{n, k}, p, X, Y) where {n, k}
    # This version is Benchmarked to use only two thirds the time of the previous inner-implementation.
    # We are only finding the LU-factorization of the (2k × 2k) matrix (p' * p).

    Q = SymplecticMatrix(p, X, Y)
    I = UniformScaling(2n)

    # Perform LU-factorization before multiplication:
    p_Tp = lu(p' * p)
    return tr(X' * (I - (1/2) * Q' * p * (p_Tp \ (p')) * Q) * (Y / p_Tp))
end

function inner_mul(::SymplecticStiefel{n, k}, p, X, Y) where {n, k}
    # Not faster, and does not save allocations...

    Q = SymplecticMatrix(p, X, Y)
    I = UniformScaling(2n)
    # Perform LU-factorization before multiplication by inverse:
    p_Tp = lu(p' * p)

    # temp_2n_2k = similar(p)
    temp_2k_2k = similar(p, (2k, 2k))
    temp_2n_2n = similar(p, (2n, 2n))
    temp_2k_2n = similar(p, (2k, 2n))

    # End goal:
    # tr(X' * (I - (1/2) * Q' * p * (p_Tp \ (p')) * Q) * (Y / p_Tp))

    # Compute the inner matrix in-place as much as possible: 
    # First: (p_Tp \ p') -> temp_2k_2n
    ldiv!(temp_2k_2n, p_Tp, p')

    # Then: p * (p_Tp \ p') -> temp_2n_2n
    mul!(temp_2n_2n, p, temp_2k_2n)

    # (Q'/2)* p * (p_Tp \ p') * Q -> temp_2n_2n
    lmul!((1/2)*Q', temp_2n_2n)
    rmul!(temp_2n_2n, Q)

    inner_matrix = I + lmul!(-1, temp_2n_2n) 

    # Left-multiply:
    # Multiply X' * inner_matrix from the left -> in 'temp_2k_2n':
    mul!(temp_2k_2n, X', inner_matrix)

    # Hard to do:
    # Y * (p_Tp)^{1} -> temp_2n_2k:
    
    # Go directly to:
    # (X' * inner_matrix) * (Y * (p_Tp)^{1}) -> temp_2k_2k:
    mul!(temp_2k_2k, temp_2k_2n, Y / p_Tp)
    
    return tr(temp_2k_2k)    
end


function Base.inv(::SymplecticStiefel, p) 
    Q = SymplecticMatrix(p)
    return Q' * p' * Q 
end

#= 
Bad Idea: Nothing is square...
function inv!(::SymplecticStiefel, q, p) 
    Q = SymplecticMatrix(p)
    mul!(q, p', Q)
    lmul!(Q', q) 
    return q
end
=#

function change_representer!(::SymplecticStiefel{n, k}, Y, ::EuclideanMetric, p, X) where {n, k}
    # Quite an ugly expression: Have checked and it seems to be working.
    Q = SymplecticMatrix(p, X)
    I = UniformScaling(2n)

    # Remove memory allocation:
    # A = factorize((I - (1/2) * Q' * p * ((p' * p) \ p') * Q)')

    Y .= (lu((I - (1/2) * Q' * p * ((p' * p) \ p') * Q)') \ X) * p' * p
    return Y
end

@doc raw"""
    rand(M::SymplecticStiefel{n, k})

Use the canonical projection by first generating a random symplectic matrix of the correct size,
and then projecting onto the Symplectic Stiefel manifold.
"""
Base.rand(M::SymplecticStiefel{n, k}) where {n, k} = begin
    canonical_projection(M, rand(Symplectic(n)))
end

@doc raw"""
    rand(::SymplecticStiefel{n, k}, p)

As based on the parametrization of the tangent space ``T_p\operatorname{SpSt}(n, k)`` found in Proposition 3.2
of Benodkat-Zimmermann. There they express the tangent space as ``X = pΩ + p^sB``, where ``Ω^+ = -Ω`` is Hamiltonian.
The notation ``p^s`` means the symplectic complement of ``p`` s.t. ``p^{+}p^{s} = 0``, and ``B ∈ ℝ^{2(n-k) × 2k}.
"""
Base.rand(::SymplecticStiefel{n, k}, p) where {n, k} = begin
    Ω = rand_hamiltonian(Symplectic(k))
    p * Ω
end


@doc raw"""
    π(::SymplecticStiefel{n, k}, p) where {n, k}    

Define the canonical projection from ``\operatorname{Sp}(2n, 2n)`` onto ``\operatorname{SpSt}(2n, 2k)``,
by projecting onto the first ``k`` columns, and the ``n + 1``'th column onto the ``n + k``'th columns.

It is assumed that the point ``p`` is on ``\operatorname{Sp}(2n, 2n)``.

# As done in Bendokat Zimmermann in equation 3.2.
"""
function canonical_projection(M::SymplecticStiefel{n, k}, p) where {n, k}
    p_SpSt = similar(p, (2n, 2k))
    return canonical_projection!(M, p_SpSt, p)
end

function canonical_projection!(::SymplecticStiefel{n, k}, p_SpSt, p) where {n, k}
    p_SpSt[:, (1:k)] .= p[:, (1:k)]; 
    p_SpSt[:, (k+1:2k)] .= p[:, (n+1:n+k)]
    return p_SpSt
end

@doc raw"""
    retract!(::SymplecticStiefel, q, p, X, ::CayleyRetraction)

Define the Cayley retraction on the Symplectic Stiefel manifold.
Reduced requirements down to inverting an (2k × 2k) matrix. 
Formula due to Bendokat-Zimmermann Proposition 5.2.

# TODO: Add formula from Bendokat-Zimmermann.

# We set (t=1), regulate by the norm of the tangent vector how far to move.
"""
function retract!(M::SymplecticStiefel{n, k}, q, p, X, ::CayleyRetraction) where {n, k}
    # Define intermediate matrices for later use:
    A = inv(M, p) * X
    H = X .- p*A
    q .= -p .+ (H + 2*p) / (I - A/2 + (inv(M, H)*H)/4)
    return q
end

ManifoldsBase.default_retraction_method(::SymplecticStiefel) = CayleyRetraction()

@doc raw"""
    inverse_retract!(::SymplecticStiefel, q, p, X, ::CayleyInverseRetraction)

Compute the Cayley Inverse Retraction as in proposition 5.3 of Bendorkat & Zimmermann[^Bendokat2021].

First, recall the definition the standard symplectic matrix
``Q_{2n} = 
\begin{bmatrix}
 0    & I_n \\
-I_n  & 0
\end{bmatrix}
``
as well as the symplectic inverse for a matrix ``A ∈ ℝ^{2n × 2k},
``A^{+} = Q_{2k}^T A^T Q_{2n}``.

For ``p, q ∈ \operatorname{Sp}(2n, ℝ)``, we can then define the
inverse cayley retraction as long as the following matrices exist.
````math
    U = (I + p^+ q)^{-1}, \quad V = (I + q^+ p)^{-1}.
````

Finally, definition of the inverse cayley retration at ``p`` applied to ``q`` is
````math
\mathcal{L}_p^{\operatorname{Sp}}(q) = 2p\bigl(V - U\bigr) + 2\bigl((p + q)U - p\bigr) ∈ T_p\operatorname{Sp}(2n).   
````

[Bendokat2021]
> Bendokat, Thomas and Zimmermann, Ralf
> The real symplectic Stiefel and Grassmann manifolds: metrics, geodesics and applications
> arXiv preprint arXiv:2108.12447, 2021
"""
function inverse_retract!(M::SymplecticStiefel, X, p, q, ::CayleyInverseRetraction)
    # Speeds up solving the linear systems required for multiplication with U, V:
    U_inv = lu(I + inv(M, p) * q)
    V_inv = lu(I + inv(M, q) * p)

    X .= 2 .* ((p / V_inv .- p / U_inv) + ((p .+ q) / U_inv) .- p)
    return X
end

function gradient(M::SymplecticStiefel, f, p, backend::RiemannianProjectionBackend)
    amb_grad = _gradient(f, p, backend.diff_backend)
    return grad_euclidian_to_manifold(M, p, amb_grad)
end

function gradient!(M::SymplecticStiefel, f, X, p, backend::RiemannianProjectionBackend)
    _gradient!(f, X, p, backend.diff_backend)
    return grad_euclidian_to_manifold!(M, X, p, X)
end

function grad_euclidian_to_manifold(::SymplecticStiefel, p, ∇f_euc)
    Q = SymplecticMatrix(p, ∇f_euc)
    return (∇f_euc * p'  .+ Q * p * (∇f_euc)' * Q) * p     
end

function grad_euclidian_to_manifold!(::SymplecticStiefel, ∇f_man, p, ∇f_euc)
    Q = SymplecticMatrix(p, ∇f_euc)
    ∇f_man .= (∇f_euc * p' .+ Q * p * (∇f_euc)' * Q) * p    
    return ∇f_man
end