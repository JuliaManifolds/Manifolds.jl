@doc raw"""
    Symplectic{n, ℝ} <: AbstractEmbeddedManifold{ℝ, DefaultIsometricEmbeddingType}

Over the field ℝ, the Real Symplectic Manifold consists of all $2n × 2n$ matrices defined as 
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
``2n -> n``, as most computations with points on a Real Symplectic manifold takes advantage of the natural block structure
of a matrix ``A ∈ ℝ^{2n × 2n}`` where we consider it as consisting of four smaller matrices in ``ℝ^{n × n}``.

# Constructor:
    Symplectic(2*n, field::AbstractNumbers=ℝ) -> Symplectic{n, ℝ}()

The constructor accepts the number of dimensions in ``ℝ^{2n × 2n}`` as the embedding for the Real Symplectic manifold, 
but internally stores the integer ``n`` denoting half the dimension of the embedding. 
"""
struct Symplectic{n, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType} 
end

@doc """
    Document difference between real and complex.
    You are given a manifold of embedding dimension 2nX2n.
"""
Symplectic(n::Int, field::AbstractNumbers=ℝ) = begin 
    Symplectic{n, field}()
end

@doc """
    #TODO: Document The Riemannian Symplectic metric used.
"""
struct RealSymplecticMetric <: RiemannianMetric 
end

default_metric_dispatch(::Symplectic{n, ℝ}, ::RealSymplecticMetric) where {n, ℝ} = Val(true)

function check_point(M::Symplectic{n, ℝ}, p; kwargs...) where {n, ℝ}
    abstract_embedding_type = supertype(typeof(M))
    
    mpv = invoke(check_point, Tuple{abstract_embedding_type, typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    
    # Perform check that the matrix lives on the real symplectic manifold
    expected_identity = symplectic_inverse(M, p) * p
    p_identity = one(p)
    if !isapprox(expected_identity, p_identity; kwargs...)
        return DomainError(
            norm(expected_identity - p_identity),
            ("The point $(p) does not lie on $(M) because its symplectic" 
           * " inverse composed with itself is not the identity.")
        )
    end
    return nothing
end

# Document 'check_vector'.
@doc raw"""
    Reference: 
"""
check_vector(::Symplectic, ::Any...)

function check_vector(M::Symplectic{n}, p, X; kwargs...) where {n}
    abstract_embedding_type = supertype(typeof(M))

    mpv = invoke(
        check_vector,
        Tuple{abstract_embedding_type, typeof(p), typeof(X)},
        M, p, X;
        kwargs...,
    )
    mpv === nothing || return mpv

    # Big oopsie! (X')
    tangent_requirement_norm = norm(X' * symplectic_multiply(M, p) + p' * symplectic_multiply(M, X), 2)
    if !isapprox(tangent_requirement_norm, 0.0; kwargs...)
        return DomainError(
            tangent_requirement_norm,
            ("The matrix $(X) is not in the tangent space at point $p of the"
           * " $(M) manifold, as X'Qp + p'QX is not the zero matrix")
        )
    end
    return nothing
end

decorated_manifold(::Symplectic{n, ℝ}) where {n, ℝ} = Euclidean(2n, 2n; field=ℝ)

Base.show(io::IO, ::Symplectic{n, ℝ}) where {n, ℝ} = print(io, "Symplectic{$(2n)}()")

@doc raw"""
    symplectic_inverse(M::Symplectic{n, ℝ}, A) where {n, ℝ}

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
function symplectic_inverse(::Symplectic{n, ℝ}, A) where {n}
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
function symplectic_multiply(::Symplectic{n, ℝ}, A; left=true, transposed=false) where {n}
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
@doc raw"""
    inner(::Symplectic{n, ℝ}, p, X, Y)

Riemannian: Test Test. Reference to Fiori.

"""
function inner(M::Symplectic{n, ℝ}, p, X, Y) where {n}
    # For symplectic matrices, the 'symplectic inverse' p^+ is the actual inverse.
    p_star = symplectic_inverse(M, p)
    return tr((p_star * X)' * (p_star * Y))
end


@doc """
    grad_euclidian_to_manifold(M::Symplectic{n}, p, ∇_Euclidian_f)

Compute the transformation of the euclidian gradient of a function `f` onto the tangent space of the point p ∈ Sn(ℝ, 2n)[^FioriSimone2011].
The transformation is found by requireing that the gradient element in the tangent space solves the metric compatibility for the Riemannian default_metric_dispatch
along with the defining equation for a tangent vector ``X ∈ T_pSn(ℝ)``at a point ``p ∈ Sn(ℝ)``.    

# Could reproduce more explicit formulas?
(f needs to be defined on a neighborhood of the point p in the embedding space ℰ?)

[^FioriSimone2011]:
    > Simone Fiori:
    > Solving minimal-distance problems over the manifold of real-symplectic matrices,
    > SIAM Journal on Matrix Analysis and Applications 32(3), pp. 938-968, 2011.
    > doi [10.1137/100817115](https://doi.org/10.1137/100817115).
"""
function grad_euclidian_to_manifold(M::Symplectic, p, ∇f_euc)
    inner_expression = ∇f_euc' * symplectic_multiply(M, p; left=false) - symplectic_multiply(M, p') * ∇f_euc
    ∇f_man = (1/2) .* p * symplectic_multiply(M, inner_expression)
    return ∇f_man
end

