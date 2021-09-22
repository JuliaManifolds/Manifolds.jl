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
struct RealSymplectic{n} <: AbstractEmbeddedManifold{ManifoldsBase.ℝ, DefaultIsometricEmbeddingType} where {n}
end

function check_point(M::RealSymplectic{n}, p; kwargs...) where {n}
    p_valid = invoke(check_point, Tuple{supertype(typeof(M)), typeof(p)}, M, p; kwargs...)
    p_valid === nothing || return p_valid
    
    # Perform check that the matrix lives on the real symplectic manifold
    
end


@doc raw"""
    symplectic_inverse(A)

Apply the symplectic inverse $A^+$ to matrix 
````math 
A ∈ ℝ^{2n × 2n},\quad 
A = 
\begin{bmatrix}
A_{1,1} & A_{1,2} \\
A_{2,1} & A_{2, 2}
\end{bmatrix}
````
inplace, with the symplectic inverse defined as:
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

As a result A is transformed to:
````math
A^{+} = 
\begin{bmatrix}
[ A_{2, 2}^T & -A_{1, 2}^T \\
 -A_{2, 1}^T &  A_{2, 2}^T ]
\end{bmatrix}
````
"""
function symplectic_inverse!(A)
    # Check that A is an even dimension, square matrix. 
    two_n = LinearAlgebra.checksquare(A)
    two_n % 2 == 0 || throw(DomainError(size(A), ("The size of matrix $A must be of type " *
                                                 "(2n, 2n), n ∈ ℕ, not $(size(A))."))) 
    n = div(two_n, 2)
    # Allocate temporary storage for block matrices in A:
    block_storage = zeros(eltype(A), (n, n))
    
    # Switch and transpose block diagonals:
    block_storage[:, :] = A[1:n, 1:n]  # Store top left diagonal block.
    
    A[1:n, 1:n] = (A[(n+1):2n, (n+1):2n])'
    A[(n+1):2n, (n+1):2n] = block_storage'

    # Invert sign and transpose off-diagonal blocks:
    A[1:n, (n+1):2n] = (-1) .* A[1:n, (n+1):2n]'
    A[(n+1):2n, 1:n] = (-1) .* A[(n+1):2n, 1:n]'
    return nothing
end