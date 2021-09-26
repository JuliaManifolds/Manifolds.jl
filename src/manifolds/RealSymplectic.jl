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
    p_valid = invoke(check_point, Tuple{supertype(typeof(M)), typeof(p)}, M, p; kwargs...)
    p_valid === nothing || return p_valid
    
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
    two_n = LinearAlgebra.checksquare(A)
    two_n % 2 == 0 || throw(DomainError(size(A), ("The size of matrix $A must be of type " *
                                                 "(2n, 2n), n ∈ ℕ, not $(size(A))."))) 
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