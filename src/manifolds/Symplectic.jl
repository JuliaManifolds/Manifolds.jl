@doc raw"""
    abstract type AbstractSymplectic{n, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}

This abstract describes the base type for symplectic manifolds with reference to a specific field, and 
can thus be extended to work over both real and complex fields.
"""
abstract type AbstractSymplectic{n, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}
end

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
struct RealSymplectic{n} <: AbstractSymplectic{n, ℝ} 
end

@doc """
    #TODO: 
"""
RealSymplectic(embedding_dimension::Int) = begin 
    @assert embedding_dimension % 2 == 0; RealSymplectic{div(embedding_dimension, 2)}() 
end

@doc """
    #TODO: Document The Riemannian Symplectic metric used.
"""
struct RealSymplecticMetric <: RiemannianMetric 
end

default_metric_dispatch(::RealSymplectic, ::RealSymplecticMetric) = Val(true)

function check_point(M::RealSymplectic{n}, p; kwargs...) where {n}
    # This 'nested' supertype call was needed to avoid StackOverflow (the error, not the website :P ) 
    # Does not look very pretty though, is there any prettier way to achieve the desired levels of abstraction, 
    # whilst keeping the advantages from using an 'AbstractEmbeddedManifold' as the supertype?
    abstract_embedding_type = supertype(typeof(M))
    
    mpv = invoke(check_point, Tuple{abstract_embedding_type, typeof(p)}, M, p; kwargs...)
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

# Document 'check_vector'.
@doc raw"""
    Reference: 
"""
check_vector(::RealSymplectic, ::Any...)

function check_vector(M::RealSymplectic{n}, p, X; kwargs...) where {n}
    # This 'nested' supertype call was needed to avoid StackOverflow (the error, not the website :P ) 
    # Does not look very pretty though, is there any prettier way to achieve the desired levels of abstraction, 
    # whilst keeping the advantages from using an 'AbstractEmbeddedManifold' as the supertype?
    abstract_embedding_type = supertype(typeof(M))

    mpv = invoke(
        check_vector,
        Tuple{abstract_embedding_type, typeof(p), typeof(X)},
        M, p, X;
        kwargs...,
    )
    mpv === nothing || return mpv

    tangent_requirement_norm = norm(X * symplectic_multiply(M, p) + p' * symplectic_multiply(M, X), 2)
    if !isapprox(tangent_requirement_norm, 0.0, kwargs...)
        return DomainError(
            tangent_requirement_norm,
            ("The matrix $(X) is not in the tangent space at point $p of the"
           * " RealSymplectic($(2n)) manifold, as X'Qp + p'QX is not the zero matrix")
        )
    end
    return nothing
end

decorated_manifold(::RealSymplectic{n}) where {n} = Euclidean(2n, 2n; field=ℝ)

Base.show(io::IO, ::RealSymplectic{n}) where {n} = print(io, "RealSymplectic{$(2n)}()")

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

# TODO: implement logarithmic map.
# DOME: Implement internally as storing the 'n' of the '2n' dimensions embebbed.
