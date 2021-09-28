@doc raw"""
    RealSymplectic{N} <: AbstractEmbeddedManifold{ℝ} where {N}

The Real Symplectic Manifold consists of all $2n × 2n$ matrices defined as 

````math
\operatorname{Sp}(2n, ℝ) = \bigl\{ p ∈ ℝ^{2n × 2n} \, \big| \, p^TQ_{2n}p = Q_{2n} \bigr\}
```` 
where 
````math
Q_{2n} = 
\begin{bmatrix}
[0_n & I_n \\
 -I_n & 0_n ]
\end{bmatrix}
```` 
with $0_n$ and $I_n$ denoting the $n × n$ zero-matrix and indentity matrix respectively.

# Constructor:
    RealSymplectic(n)
"""
struct RealSymplectic{n, ℝ} <: AbstractEmbeddedManifold{ℝ, DefaultIsometricEmbeddingType}
end

RealSymplectic(n::Int, field::AbstractNumbers=ℝ) = RealSymplectic{n, ℝ}()

function check_point(M::RealSymplectic{n}, p; kwargs...) where {n}
    mpv = invoke(check_point, Tuple{supertype(typeof(M)), typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    
    # Perform check that the matrix lives on the real symplectic manifold
    expected_identity = symplectic_inverse(p) * p
    p_identity = one(p)
    if !isapprox(expected_identity, p_identity, kwargs...)
        return DomainError(
            norm(expected_identity - p_identity),
            "The point $(p) does not lie on $(M) because its symplectic inverse composed with itself is not the identity."
        )
    end
    return nothing
end

function check_vector(M::RealSymplectic{n}, p, X; kwargs...) where {n}
    mpv = invoke(
        check_vector,
        Tuple{supertype(typeof(M)), typeof(p), typeof(X)},
        M, p, X;
        kwargs...,
    )
    mpv === nothing || return mpv

    tangent_requirement_norm = norm(X * symplectic_multiply(p) + p' * symplectic_multiply(X), 2)
    if !isapprox(tangent_requirement_norm, 0.0, kwargs...)
        return DomainError(
            tangent_requirement_norm,
            "The matrix $(X) is not in the tangent space at point $p of the RealSymplectic($(2n)) manifold , as X'Qp + p'QX is not the zero matrix"
        )
    end
    return nothing
end

decorated_manifold(::RealSymplectic{N,𝔽}) where {N,𝔽} = Euclidean(N, N; field=𝔽)

@doc raw"""
    symplectic_inverse(A)

Compute the symplectic inverse $A^+$ of matrix A, returning the result.
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
[0_n & I_n \\
 -I_n & 0_n ]
\end{bmatrix}
````

In total the symplectic inverse of A is:
````math
A^{+} = 
\begin{bmatrix}
[ A_{2, 2}^T & -A_{1, 2}^T \\
 -A_{2, 1}^T &  A_{2, 2}^T ]
\end{bmatrix}
````
"""
function symplectic_inverse(A)
    # Check that A is of an even dimension, square matrix. 
    two_n = check_even_dimension_square(A)
    n = div(two_n, 2)

    # Allocate memory for A_star, the symplectic inverse:
    A_star = zeros(eltype(A), (two_n, two_n))
    
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
function symplectic_multiply(A; left=true, transposed=false)
    two_n = check_even_dimension_square(A)
    n = div(two_n, 2)

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
