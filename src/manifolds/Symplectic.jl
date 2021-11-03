@doc raw"""
    Symplectic{n, ℝ} <: AbstractEmbeddedManifold{ℝ, DefaultIsometricEmbeddingType}

Over the field ℝ, the Symplectic Manifold consists of all $2n × 2n$ matrices defined as
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

decorated_manifold(::Symplectic{n, ℝ}) where {n} = Euclidean(2n, 2n; field=ℝ)

@doc raw"""
    manifold_dimension(::Symplectic{n})

As a special case of the SymplecticStiefel manifold with k = n. As shown in Proposition
3.1 in Gao et. al.
"""
manifold_dimension(::Symplectic{n}) where {n} = (2n + 1)*n

Base.show(io::IO, ::Symplectic{n, ℝ}) where {n, ℝ} = print(io, "Symplectic{$(2n)}()")

@doc raw"""
    #TODO: Document The Riemannian Symplectic metric used.

````math
    g_p(Z_1, Z_2) = tr((p^{-1}Z_1)^T (p^{-1}Z_2))
````
"""
struct RealSymplecticMetric <: RiemannianMetric
end

default_metric_dispatch(::Symplectic{n, ℝ}, ::RealSymplecticMetric) where {n, ℝ} = Val(true)

function check_point(M::Symplectic{n, ℝ}, p; kwargs...) where {n, ℝ}
    abstract_embedding_type = supertype(typeof(M))

    mpv = invoke(check_point, Tuple{abstract_embedding_type, typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv

    # Perform check that the matrix lives on the real symplectic manifold:
    expected_zero = norm(inv(M, p) * p - LinearAlgebra.I)
    if !isapprox(expected_zero, zero(eltype(p)); kwargs...)
        return DomainError(
            expected_zero,
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

    Q = SymplecticMatrix(p, X)
    tangent_requirement_norm = norm(X' * Q * p + p' * Q * X, 2)

    if !isapprox(tangent_requirement_norm, 0.0; kwargs...)
        return DomainError(
            tangent_requirement_norm,
            ("The matrix $(X) is not in the tangent space at point $p of the"
           * " $(M) manifold, as X'Qp + p'QX is not the zero matrix")
        )
    end
    return nothing
end

Base.inv(M::Symplectic, p) = symplectic_inverse(M, p)

Base.rand(M::Symplectic{n}) where {n} = begin
    # Generate random matrices to construct a Hamiltonian matrix:
    Ω = rand_hamiltonian(M)
    (I + Ω) / (I - Ω)
end

@doc raw"""
    inner(::Symplectic{n, ℝ}, p, X, Y)

Riemannian: Test Test. Reference to Fiori.

"""
function inner(M::Symplectic{n, ℝ}, p, X, Y) where {n}
    # For symplectic matrices, the 'symplectic inverse' p^+ is the actual inverse.
    p_star = inv(M, p)
    return tr((p_star * X)' * (p_star * Y))
end



@doc raw"""
    check_even_dim(p; square=false)::Integer

Convenience function to check whether or not an abstract matrix is square, with an even
number (2n, 2n) of rows and columns. Then returns the integer part of the even dimension.
"""
function check_even_dim(p; square=false)
    n, k = size(p)
    # Otherwise, both dimensions just need to be even.
    # First check that dimensions are even:
    ((n % 2 == 0) && (k % 2 == 0)) || throw(DimensionMismatch("Matrix does not have even "
                                      * "dimensions (2n, 2k): Dimensions are ($(n), $(k))."))

    # If 'square=true', we require m==n:
    (!square || (n == k)) || throw(DimensionMismatch("Matrix is not square with dimensions "
                                                   * "(2n, 2n): Dimensions are ($(n), $(k))."))

    return div(n, 2), div(k, 2)
end

# T Indicates whether or not transposed.
# Acts like the symplectic transform.
struct SymplecticMatrix{T}
    λ::T
end
SymplecticMatrix(λ::T) where {T <: Number} = SymplecticMatrix{T}(λ)

SymplecticMatrix(arrays::Vararg{AbstractArray}) = begin
    TS = Base.promote_type(map(eltype, arrays)...)
    SymplecticMatrix(one(TS))
end

ndims(Q::SymplecticMatrix) = 2
copy(Q::SymplecticMatrix) = SymplecticMatrix(Q.λ)
Base.eltype(::SymplecticMatrix{T}) where {T} = T
Base.convert(::Type{SymplecticMatrix{T}}, Q::SymplecticMatrix) where {T} = SymplecticMatrix(convert(T, Q.λ))

function Base.show(io::IO, Q::SymplecticMatrix)
    s = "$(Q.λ)"
    if occursin(r"\w+\s*[\+\-]\s*\w+", s)
        s = "($s)"
    end
    print(io, typeof(Q), "(): $(s)*[0 I; -I 0]")
end

# Overloaded functions:
# Overload: * scalar left, right, matrix, left, right, itself right and left.
# unary -, inv = -1/s,
# transpose = -s, +.

(Base.:-)(Q::SymplecticMatrix) = SymplecticMatrix(-Q.λ)

(Base.:*)(x::Number, Q::SymplecticMatrix) = SymplecticMatrix(x*Q.λ)
(Base.:*)(Q::SymplecticMatrix, x::Number) = SymplecticMatrix(x*Q.λ)
(Base.:*)(Q1::SymplecticMatrix, Q2::SymplecticMatrix) = LinearAlgebra.UniformScaling(-Q1.λ*Q2.λ)

Base.transpose(Q::SymplecticMatrix) = -Q
Base.adjoint(Q::SymplecticMatrix) = -Q
Base.inv(Q::SymplecticMatrix) = SymplecticMatrix(-(1/Q.λ))

(Base.:+)(Q1::SymplecticMatrix, Q2::SymplecticMatrix) = SymplecticMatrix(Q1.λ + Q2.λ)
(Base.:-)(Q1::SymplecticMatrix, Q2::SymplecticMatrix) = SymplecticMatrix(Q1.λ - Q2.λ)

(Base.:+)(Q::SymplecticMatrix, p::AbstractMatrix) = p + Q
function (Base.:+)(p::AbstractMatrix, Q::SymplecticMatrix)
    # When we are adding, the Matrices must match in size:
    n, _ = check_even_dim(p; square=true)

    # Allocate new memory:
    TS = Base._return_type(+, Tuple{eltype(p), eltype(Q)})
    out = copyto!(similar(p, TS), p)

    # Add Q.λ multiples of the UniformScaling to the lower left and upper right blocks of p:
    λ_Id = LinearAlgebra.UniformScaling(Q.λ)

    out[1:n, (n+1):2n] += λ_Id
    out[(n+1):2n, 1:n] -= λ_Id
    return out
end

# Binary minus:
(Base.:-)(Q::SymplecticMatrix, p::AbstractMatrix) = Q + (-p)
(Base.:-)(p::AbstractMatrix, Q::SymplecticMatrix) = p + (-Q)

function (Base.:*)(p::AbstractMatrix, Q::SymplecticMatrix)
    _, k = check_even_dim(p)

    # Allocate new memory:
    TS = Base._return_type(+, Tuple{eltype(p), eltype(Q)})
    pQ = similar(p, TS)

    # Perform right mulitply by λ*Q:
    pQ[:, 1:k] = (-Q.λ).*p[:, (k+1):end]
    pQ[:, (k+1):end] = (Q.λ) .*p[:, 1:k]

    return pQ
end

function (Base.:*)(Q::SymplecticMatrix, p::AbstractMatrix)
    n, _ = check_even_dim(p)

    # Allocate new memory:
    TS = Base._return_type(+, Tuple{eltype(p), eltype(Q)})
    Qp = similar(p, TS)

    # Perform left mulitply by λ*Q:
    Qp[1:n, :] = (Q.λ) .* p[(n+1):end, :]
    Qp[(n+1):end, :] = (-Q.λ) .* p[1:n, :]

    return Qp
end


function LinearAlgebra.lmul!(Q::SymplecticMatrix, p::AbstractMatrix)
    # Perform left multiplication by a symplectic matrix,
    # overwriting the matrix p in place:
    n, k = check_even_dim(p)

    # Need to allocate half the space in order to avoid overwriting:
    TS = Base._return_type(+, Tuple{eltype(p), eltype(Q)})
    half_row_p = similar(p, TS, (n, 2k))
    half_row_p[1:n, :] = p[1:n, :]

    # Perform left mulitply by λ*Q:
    p[1:n, :] = (Q.λ) .* p[(n+1):end, :]
    p[(n+1):end, :] = (-Q.λ) .* half_row_p[1:n, :]

    return p
end

function LinearAlgebra.rmul!(p::AbstractMatrix, Q::SymplecticMatrix)
    # Perform right multiplication by a symplectic matrix,
    # overwriting the matrix p in place:
    n, k = check_even_dim(p)

    # Need to allocate half the space in order to avoid overwriting:
    TS = Base._return_type(+, Tuple{eltype(p), eltype(Q)})
    half_col_p = similar(p, TS, (2n, k))
    half_col_p[:, 1:k] = p[:, 1:k]

    # Allocate new memory:
    TS = Base._return_type(+, Tuple{eltype(p), eltype(Q)})

    # Perform right mulitply by λ*Q:
    p[:, 1:k] = (-Q.λ).*p[:, (k+1):end]
    p[:, (k+1):end] = (Q.λ) .*half_col_p[:, 1:k]

    return p
end

function LinearAlgebra.mul!(A::AbstractMatrix, p::AbstractMatrix, Q::SymplecticMatrix)
    _, k = check_even_dim(p)
    # Perform right mulitply by λ*Q:
    A[:, 1:k] = (-Q.λ).*p[:, (k+1):end]
    A[:, (k+1):end] = (Q.λ) .* half_col_p[:, 1:k]
    return A
end

function LinearAlgebra.mul!(A::AbstractMatrix, Q::SymplecticMatrix, p::AbstractMatrix)
    n, _ = check_even_dim(p)
    # Perform right mulitply by λ*Q:
    A[1:n, :] = (Q.λ) .* p[(n+1):end, :]
    A[(n+1):end, :] = (-Q.λ) .* half_row_p[1:n, :]
    return A
end

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

function rand_hamiltonian(::Symplectic{n}; final_norm=1) where {n}
    A = randn(n, n); B = randn(n, n); C = randn(n, n)
    B = (1/2) .* (B .+ B'); C = (1/2) .* (C .+ C')
    Ω = [A B; C -A']
    return final_norm*Ω/norm(Ω, 2)
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


@doc raw"""
    grad_euclidian_to_manifold(M::Symplectic{n}, p, ∇_Euclidian_f)

Compute the transformation of the euclidian gradient of a function `f` onto the tangent space of the point p ∈ Sn(ℝ, 2n)[^FioriSimone2011].
The transformation is found by requireing that the gradient element in the tangent space solves the metric compatibility for the Riemannian default_metric_dispatch
along with the defining equation for a tangent vector ``X ∈ T_pSn(ℝ)``at a point ``p ∈ Sn(ℝ)``.

First we change the representation of the gradient from the Euclidean metric to the RealSymplecticMetric at p,
and then we project the result onto the tangent space ``T_p\operatorname{Sp}(2n, ℝ)`` at p.

[^FioriSimone2011]:
    > Simone Fiori:
    > Solving minimal-distance problems over the manifold of real-symplectic matrices,
    > SIAM Journal on Matrix Analysis and Applications 32(3), pp. 938-968, 2011.
    > doi [10.1137/100817115](https://doi.org/10.1137/100817115).
"""
function grad_euclidian_to_manifold(M::Symplectic{n}, p, ∇f_euc) where {n}
    ∇f_metr_comp = change_representer(M, EuclideanMetric(), p, ∇f_euc)
    return project(M, p, ∇f_metr_comp)
end

# Overwrite gradient functions for the Symplectic case:
# Need to first change representer of ``∇f_euc`` to the Symplectic manifold,
# then project onto the correct tangent space.
function gradient(M::Symplectic, f, p, backend::RiemannianProjectionBackend)
    amb_grad = _gradient(f, p, backend.diff_backend)

    # Does not work: Change_representer ∘ Proj(amb_grad):
    # return change_representer(M, EuclideanMetric(), p, project(M, p, amb_grad))

    # Proj ∘ Change_representer(amb_grad):
    return project(M, p, change_representer(M, EuclideanMetric(), p, amb_grad))
end

function gradient!(M::Symplectic, f, X, p, backend::RiemannianProjectionBackend)
    _gradient!(f, X, p, backend.diff_backend)
    change_representer!(M, X, EuclideanMetric(), p, X)
    return project!(M, X, p, X)
end

@doc raw"""
    change_representer!(::Symplectic, Y, p, X)

Change the representation of a tangent vector ``χ ∈ T_p\operatorname{Sp}(2n, ℝ)`` s.t.
````math
    g_p(c(χ), η) = ⟨χ, η⟩^{\text{Euc}} \;∀\; η ∈ T_p\operatorname{Sp}(2n, ℝ).
````
s.t.

"""
function change_representer!(::Symplectic, Y, ::EuclideanMetric, p, X)
    # The following formula actually works for all X ∈ ℝ^{2n × 2n}, and
    # is exactly the same as: Proj_[T_pSp](p * p^T * X).
    # Q = SymplecticMatrix(p, X)
    # Y .= (1/2) .* p * p' * X .+ (1/2) .* p * Q * X' * p * Q
    # The above is also the only formula I have found for a 'change_representer' which
    # stays in the tangent space of p after application.

    Y .= p * p' * X
    return Y
end


@doc raw"""
    project!(M::Symplectic{n, ℝ}, Y, p, X) where {n}

Compute the projection of ``X ∈ R^{2n × 2n}`` onto ``T_p\operatorname{Sp}(2n, ℝ)``, stored inplace in Y.
Adapted from projection onto tangent spaces of Symplectic Stiefal manifolds ``\operatorname{Sp}(2p, 2n)`` with
``p = n``[^Gao2021riemannian].

# Full defining equations possibly:

[^Gao2021riemannian]:
    > Gao, Bin and Son, Nguyen Thanh and Absil, P-A and Stykel, Tatjana:
    > Riemannian optimization on the symplectic Stiefel manifold,
    > SIAM Journal on Optimization 31(2), pp. 1546-1575, 2021.
    > doi [10.1137/20M1348522](https://doi.org/10.1137/20M1348522)
"""
function project!(::Symplectic{n, ℝ}, Y, p, X) where {n}
    # Original formulation of the projection from the Gao et al. paper:
    # pT_QT = symplectic_multiply(M, p'; left=false, transposed=true)
    # Y[:, :] = pQ * symmetrized_pT_QT_X .+ (I - pQ*pT_QT) * X
    # The term: (I - pQ*pT_QT) = 0 in our symplectic case.

    # pQ = symplectic_multiply(M, p; left=false)
    # pT_QT_X = symplectic_multiply(M, p'; left=false, transposed=true) * X

    # TS = Base._return_type(+, Tuple{eltype(p), eltype(X)})
    Q = SymplecticMatrix(p, X)

    pT_QT_X = p' * Q' * X
    symmetrized_pT_QT_X = (1/2) .* (pT_QT_X + pT_QT_X')

    Y[:, :] = p*Q*(symmetrized_pT_QT_X)
    return Y
end


@doc raw"""
    project_normal!(M::Symplectic{n, ℝ}, Y, p, X)


Project onto the normal space relative to the tangent space at a point ``p ∈ \operatorname{Sp}(2n)``, as found in Gao et al.[^Gao2021riemannian].

# Defining equations:

[^Gao2021riemannian]:
    > Gao, Bin and Son, Nguyen Thanh and Absil, P-A and Stykel, Tatjana:
    > Riemannian optimization on the symplectic Stiefel manifold,
    > SIAM Journal on Optimization 31(2), pp. 1546-1575, 2021.
    > doi [10.1137/20M1348522](https://doi.org/10.1137/20M1348522)
"""
function project_normal!(M::Symplectic{n, ℝ}, Y, p, X) where {n}
    Q = SymplecticMatrix(p, X)

    pT_QT_X = p' * Q' * X
    skew_pT_QT_X = (1/2) .* (pT_QT_X .- pT_QT_X')

    Y[:, :] = p*Q*skew_pT_QT_X
    return Y
end


### TODO: implement retractions. First up, Cauchy-retraction:
@doc raw"""
    retract(::Symplectic, p, X, ::CayleyRetraction)

Compute the Cayley retraction on ``p ∈ \operatorname{Sp}(2n, ℝ)`` in the direction of tangent vector
``X ∈ T_p\operatorname{Sp}(2n, ℝ)``.

Defined pointwise as
````math
\mathcal{R}_p(X) = -p(p^TQ^T X + 2X)^{-1}(p^TQ^T X - 2Q)
````
"""
function retract!(::Symplectic, q, p, X, ::CayleyRetraction)
    Q = SymplecticMatrix(p, X)

    # pTQT_X = symplectic_multiply(M, p'; left=false, transposed=true)*X
    pT_QT_X = p' * Q' * X

    q .= -p * ((pT_QT_X + 2*Q) \ (pT_QT_X - 2*Q))
    return q
end

ManifoldsBase.default_retraction_method(::Symplectic) = CayleyRetraction()


struct CayleyInverseRetraction <: AbstractInverseRetractionMethod end


# Inverse-retract:
# TODO: Write as a special case of the inverse-cayley retraction for the SymplecticStiefel case?
@doc raw"""
    inverse_retract!(M::Symplectic, X, p, q, ::CayleyInverseRetraction)

Compute the Cayley Inverse Retraction as in proposition 5.3 of Bendorkat & Zimmermann[^Bendokat2021].

First, recall the definition the standard symplectic matrix
``Q =
\begin{bmatrix}
 0    & I_n \\
-I_n  & 0
\end{bmatrix}
``
as well as the symplectic inverse ``A^{+} = Q^T A^T Q``.

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
function inverse_retract!(M::Symplectic, X, p, q, ::CayleyInverseRetraction)
    # Speeds up solving the linear systems required for multiplication with U, V:
    U_inv = lu(I + inv(M, p) * q)
    V_inv = lu(I + inv(M, q) * p)

    X .= 2 .* ((p / V_inv .- p / U_inv) + ((p .+ q) / U_inv) .- p)
    return X
end

# Check vector, check point For Symplectic Stiefel.
# Retract, Inverse-retract for Stiefel.
