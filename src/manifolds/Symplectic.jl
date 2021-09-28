@doc raw"""
    Symplectic{N, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}

This structure describes the symplectic manifolds without reference to a specific field, and 
can thus be extended to work over both real and complex fields.

# Constructor:
    Symplectic(n, 𝔽) -> Symplectic{n, 𝔽}()
"""
abstract type Symplectic{n, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}
end

Symplectic(n::Int, 𝔽::AbstractNumbers) = Symplectic{n, 𝔽}()

@doc raw"""
    RealSymplectic{N} <: Symplectic{N, ℝ} where {N}

The Real Symplectic Manifold consists of all $2n × 2n$ matrices defined as 

````math
\operatorname{Sp}(2n, ℝ) = \bigl\{ p ∈ ℝ^{2n × 2n} \, \big| \, p^TQ_{2n}p = Q_{2n} \bigr\}
```` 
where 
````math
Q_{2n} = 
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n 
\end{bmatrix}
```` 
with $0_n$ and $I_n$ denoting the $n × n$ zero-matrix and indentity matrix respectively. 
This way of embedding a symplectic manifold in a real matrix space with twice the dimensions 
along the rows and columns can be seen the 'realification' of an underlying complex structure. 
Internally the dimensionality of the structure is stored as half of the even dimension supplied to the constructor, 
``2n -> n``, as most computations with points on a RealSymplectic manifold takes advantage of the natural block structure
of a matrix ``A ∈ ℝ^{2n × 2n}`` where we consider it as consisting of four smaller matrices in ``ℝ^{n × n}``.

# Constructor:
    RealSymplectic(two_n) -> Symplectic{two_n/2, ℝ}

The constructor accepts the number of dimensions in ``ℝ^{2n × 2n}`` as the embedding for the RealSymplectic manifold, 
but internally stores the integer ``n`` denoting half the dimension of the embedding. 
"""
struct RealSymplectic{n} <: Symplectic{n, ℝ} where {n}
end

RealSymplectic(two_n::Int) = begin @assert two_n % 2 == 0; Symplectic{div(two_n, 2), ℝ}() end

function check_point(M::RealSymplectic{n}, p; kwargs...) where {n}
    mpv = invoke(check_point, Tuple{supertype(typeof(M)), typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    
    # Perform check that the matrix lives on the real symplectic manifold
    expected_identity = symplectic_inverse(M, p) * p
    p_identity = one(p)
    if !isapprox(expected_identity, p_identity, kwargs...)
        return DomainError(
            norm(expected_identity - p_identity),
            "The point $(p) does not lie on $(M) because its symplectic inverse composed with itself is not the identity."
        )
    end
    return nothing
end

# Document 'is_vector'.
@doc raw"""
    Reference: 
"""
check_point(::RealSymplectic, ::Any...)

function check_vector(M::RealSymplectic{n}, p, X; kwargs...) where {n}
    mpv = invoke(
        check_vector,
        Tuple{supertype(typeof(M)), typeof(p), typeof(X)},
        M, p, X;
        kwargs...,
    )
    mpv === nothing || return mpv

    tangent_requirement_norm = norm(X * symplectic_multiply(M, p) + p' * symplectic_multiply(M, X), 2)
    if !isapprox(tangent_requirement_norm, 0.0, kwargs...)
        return DomainError(
            tangent_requirement_norm,
            "The matrix $(X) is not in the tangent space at point $p of the RealSymplectic($(2n)) manifold , as X'Qp + p'QX is not the zero matrix"
        )
    end
    return nothing
end

decorated_manifold(::RealSymplectic{N}) where {N} = Euclidean(2N, 2N; field=ℝ)

@doc raw"""
    symplectic_inverse(M::RealSymplectic{n}, A) where {n}

Compute the symplectic inverse ``A^+`` of matrix ``A ∈ ℝ^{2n × 2n}``, returning the result.
````math 
A ∈ ℝ^{2n × 2n},\quad 
A = 
\begin{bmatrix}
A_{1,1} & A_{1,2} \\
A_{2,1} & A_{2, 2}
\end{bmatrix}
````
Here the symplectic inverse is defined as:
````math
A^{+} := Q_{2n}^T A^T Q_{2n}
````
where 
````math
Q_{2n} = 
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n 
\end{bmatrix}
````

In total the symplectic inverse of A is computed as:
````math
A^{+} = 
\begin{bmatrix}
 A_{2, 2}^T & -A_{1, 2}^T \\
 -A_{2, 1}^T &  A_{2, 2}^T 
\end{bmatrix}
````
"""
function symplectic_inverse(::RealSymplectic{n}, A) where {n}
    # Allocate memory for A_star, the symplectic inverse:
    A_star = similar(A)
    
    A_star[1:n, 1:n] = (A[(n+1):2n, (n+1):2n])'
    A_star[(n+1):2n, (n+1):2n] = (A[1:n, 1:n])'

    # Invert sign and transpose off-diagonal blocks:
    A_star[1:n, (n+1):2n] = (-1) .* A[1:n, (n+1):2n]'
    A_star[(n+1):2n, 1:n] = (-1) .* A[(n+1):2n, 1:n]'
    return A_star
end

@doc raw"""
    TODO:
"""
function symplectic_multiply(::RealSymplectic{n}, A; left=true, transposed=false) where {n}
    # Flip sign if the Q-matrix to be multiplied with A is transposed:
    sign = transposed ? (-1.0) : (1.0) 

    QA = similar(A)
    if left  # Perform left multiplication by Q
        QA[1:n, :] = sign.*A[(n+1):end, :] 
        QA[(n+1):end, :] = (-sign).*A[1:n, :]
    else     # Perform right multiplication by Q
        QA[:, 1:n] = (-sign).*A[:, (n+1):end]
        QA[:, (n+1):end] = sign.*A[:, 1:n]
    end
    return QA
end

@doc """Deprecated in favor of more explicit Manifold typing."""
function check_even_dimension_square(A)
    # Check that A is an even dimension, square, matrix. 
    two_n = LinearAlgebra.checksquare(A)
    two_n % 2 == 0 || throw(DomainError(size(A), 
                            ("The size of matrix $A must be of type " *
                             "(2n, 2n), n ∈ ℕ to apply symplectic operations, not $(size(A))."))
                            ) 
    return two_n
end

function exp!(::RealSymplectic, q, p, X)
    p_inv = inv(p)
    q .= p*LinearAlgebra.exp(p_inv * X)
end

# implement Pseudo-Riemannian metric as subtyupe of AbstracMetric, look at SPD-s.
# implement logarithmic map.
# Implement internally as storing the 'n' of the '2n' dimensions embebbed.
