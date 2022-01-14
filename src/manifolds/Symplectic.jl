@doc raw"""
    Symplectic{n, ℝ} <: AbstractEmbeddedManifold{ℝ, DefaultIsometricEmbeddingType}

Over the real number field ℝ the elements of the Symplectic Manifold
are all $2n × 2n$ matrices satisfying the requirement
````math
\operatorname{Sp}(2n, ℝ) = \bigl\{ p ∈ ℝ^{2n × 2n} \, \big| \, p^TQ_{2n}p = Q_{2n} \bigr\},
````
where
````math
Q_{2n} =
\begin{bmatrix}
  0_n & I_n \\
 -I_n & 0_n
\end{bmatrix},
````
with $0_n$ and $I_n$ denoting the $n × n$ zero-matrix
and indentity matrix in ``ℝ^{n \times n}`` respectively.

Internally the dimensionality of the structure is stored as half of the even dimension
supplied to the constructor, `Symplectic(2n) -> Symplectic{n}()`,
as most computations with points on the Real Symplectic manifold takes advantage of the
natural block structure
of a matrix ``A ∈ ℝ^{2n × 2n}`` where we consider it as consisting of four
smaller matrices in ``ℝ^{n × n}``.
"""
struct Symplectic{n,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

@doc """
    Symplectic(n, field::AbstractNumbers=ℝ) -> Symplectic{div(n, 2), ℝ}()

# Constructor:
The constructor for the [`Symplectic`](@ref) manifold accepts the even embedding
dimension ``n = 2k`` for the real symplectic manifold, ``ℝ^{2k × 2k}``.
"""
function Symplectic(n::Int, field::AbstractNumbers=ℝ)
    n % 2 == 0 || throw(ArgumentError("The dimensionality of the symplectic manifold
                        embedding space must be even. Was odd, n % 2 == $(n % 2)."))
    Symplectic{div(n, 2),field}()
end

decorated_manifold(::Symplectic{n,ℝ}) where {n} = Euclidean(2n, 2n; field=ℝ)

@doc raw"""
    manifold_dimension(::Symplectic{n})

Returns the dimension of the symplectic manifold embedded in ``ℝ^{2n \times 2n}``,
i.e.
````math
    \operatorname{dim}(\operatorname{Sp}(2n)) = (2n + 1)n.
````
"""
manifold_dimension(::Symplectic{n}) where {n} = (2n + 1) * n

Base.show(io::IO, ::Symplectic{n,ℝ}) where {n,ℝ} = print(io, "Symplectic{$(2n)}()")

@doc raw"""
The canonical Riemannian metric on the symplectic manifold,
defined pointwise for ``p \in \operatorname{Sp}(2n)`` by [^FioriSimone2011]
````math
\begin{align*}
    & g_p \colon T_p\operatorname{Sp}(2n) \times T_p\operatorname{Sp}(2n) \rightarrow ℝ, \\
    & g_p(Z_1, Z_2) = \operatorname{tr}((p^{-1}Z_1)^T (p^{-1}Z_2)).
\end{align*}
````
This metric is also the default metric used within `Maniolds.jl`.
"""
struct RealSymplecticMetric <: RiemannianMetric end

default_metric_dispatch(::Symplectic{n,ℝ}, ::RealSymplecticMetric) where {n,ℝ} = Val(true)

@doc raw"""
    check_point(M::Symplectic, p; kwargs...)

Check whether `p` is a valid point on the [`Symplectic`](@ref) `M`=$\operatorname{Sp}(2n)$,
i.e. that it has the right [`AbstractNumbers`](@ref) type and $p^{+}p$ is (approximately)
the identity, where $A^{+} = Q_{2n}^TA^TQ_{2n}$ is the symplectic inverse, with
````math
Q_{2n} =
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n
\end{bmatrix}.
````
The tolerance can be set with `kwargs...` (e.g. `atol = 1.0e-14`).
"""
function check_point(M::Symplectic{n,ℝ}, p; kwargs...) where {n,ℝ}
    abstract_embedding_type = supertype(typeof(M))

    mpv = invoke(check_point, Tuple{abstract_embedding_type,typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv

    # Perform check that the matrix lives on the real symplectic manifold:
    expected_zero = norm(inv(M, p) * p - LinearAlgebra.I)
    if !isapprox(expected_zero, zero(eltype(p)); kwargs...)
        return DomainError(
            expected_zero,
            (
                "The point p does not lie on $(M) because its symplectic" *
                " inverse composed with itself is not the identity."
            ),
        )
    end
    return nothing
end

# Document 'check_vector'.
@doc raw"""
    check_vector(M::Symplectic, p, X; kwargs...)

Checks whether `X` is a valid tangent vector at `p` on the [`Symplectic`](@ref)
`M`=$\operatorname{Sp}(2n)$, i.e. the [`AbstractNumbers`](@ref) fits and
it (approximately) holds that $p^{T}Q_{2n}X + X^{T}Q_{2n}p = 0$,
where
````math
Q_{2n} =
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n
\end{bmatrix}.
````
The tolerance can be set with `kwargs...` (e.g. `atol = 1.0e-14`).
"""
check_vector(::Symplectic, ::Any...)

function check_vector(M::Symplectic{n}, p, X; kwargs...) where {n}
    abstract_embedding_type = supertype(typeof(M))

    mpv = invoke(
        check_vector,
        Tuple{abstract_embedding_type,typeof(p),typeof(X)},
        M,
        p,
        X;
        kwargs...,
    )
    mpv === nothing || return mpv

    Q = SymplecticMatrix(p, X)
    tangent_requirement_norm = norm(X' * Q * p + p' * Q * X, 2)

    if !isapprox(tangent_requirement_norm, 0.0; kwargs...)
        return DomainError(
            tangent_requirement_norm,
            (
                "The matrix X is not in the tangent space at point p of the" *
                " manifold $(M), as X'Qp + p'QX is not the zero matrix."
            ),
        )
    end
    return nothing
end

function Base.rand(M::Symplectic{n}, hamiltonian_norm::Number=1.0) where {n}
    # Generate random matrices to construct a Hamiltonian matrix:
    Ω = rand_hamiltonian(M; final_norm=hamiltonian_norm)
    # Return 'cay(Ω)':
    return (I - Ω) \ (I + Ω)
end

function Base.rand(::Symplectic{n}, p::AbstractMatrix) where {n}
    # Generate random symmetric matrix:
    S = rand(2n, 2n) .- 1 / 2
    S .= (1 / 2) .* (S + S')
    Q = SymplecticMatrix(p)
    lmul!(Q, S)
    return p * S
end

@doc raw"""
    inner(::Symplectic{n, ℝ}, p, X, Y)

Compute the canonical Riemannian inner product [`RealSymplecticMetric`](@ref)
````math
    g_p(X, Y) = \operatorname{tr}((p^{-1}X)^T (p^{-1}Y))
````
between the two tangent vectors ``X, Y \in T_p\operatorname{Sp}(2n)``.
"""
function inner(M::Symplectic{n,ℝ}, p, X, Y)::eltype(p) where {n}
    p_star = inv(M, p)
    return tr((p_star * X)' * (p_star * Y))
end

@doc raw"""
    distance(M::Symplectic{n}, p, q) where {n}

Approximate distance between two Symplectic matrices, as found
in eq. (7) of "A Riemannian-Steepest-Descent approach
for optimization of the real symplectic group."
"""
function distance(M::Symplectic{n}, p, q) where {n}
    return norm(log(symplectic_inverse_times(M, p, q)))
end

@doc raw"""
    exp(M::Symplectic, p, X)

The Exponential mapping on the Symplectic manifold with the
[`RealSymplecticMetric`](@ref) Riemannian metric.

For the point ``p \in \operatorname{Sp}(2n)`` the exponential mapping along the tangent
vector ``X \in T_p\operatorname{Sp}(2n)`` is computed as [^WangRealSymplecticGroup]
````math
    \operatorname{exp}_p(X) = p \operatorname{Exp}((p^{-1}X)^T)
                                \operatorname{Exp}([p^{-1}X - (p^{-1}X)^T]),
````
where ``\operatorname{Exp}(\cdot)`` denotes the matrix exponential.

[^WangRealSymplecticGroup]:
    > Wang, Jing and Sun, Huafei and Fiori, Simone:
    > A Riemannian-steepest-descent approach for optimization on the real symplectic group,
    > Mathematical Methods in the Applied Sciences, 41(11) pp. 4273-4286, 2018
    > doi [10.1002/mma.4890](https://doi.org/10.1002/mma.4890)
"""
function exp!(M::Symplectic{n}, q, p, X) where {n}
    p_star_X = symplectic_inverse_times(M, p, X)
    q .= p_star_X .- p_star_X'
    q .= p * exp(Array(p_star_X')) * exp(q)
    return q
end

@doc raw"""
    get_even_dims(p; square=false)::Integer

Convenience function to check whether or not an abstract matrix is square, with an even
number (2n, 2n) of rows and columns. Then returns the integer part of the even dimension.
"""
function get_even_dims(p; square=false)
    n, k = size(p)
    # Otherwise, both dimensions just need to be even.
    # First check that dimensions are even:
    ((n % 2 == 0) && (k % 2 == 0)) || throw(
        DimensionMismatch(
            "Matrix does not have even " *
            "dimensions (2n, 2k): Dimensions are ($(n), $(k)).",
        ),
    )

    # If 'square=true', we require m==n:
    (!square || (n == k)) || throw(
        DimensionMismatch(
            "Matrix is not square with dimensions " *
            "(2n, 2n): Dimensions are ($(n), $(k)).",
        ),
    )

    return div(n, 2), div(k, 2)
end

struct SymplecticMatrix{T}
    λ::T
end
SymplecticMatrix(λ::T) where {T<:Number} = SymplecticMatrix{T}(λ)

function SymplecticMatrix(arrays::Vararg{AbstractArray})
    begin
        TS = Base.promote_type(map(eltype, arrays)...)
        SymplecticMatrix(one(TS))
    end
end

ndims(Q::SymplecticMatrix) = 2
copy(Q::SymplecticMatrix) = SymplecticMatrix(Q.λ)
Base.eltype(::SymplecticMatrix{T}) where {T} = T
function Base.convert(::Type{SymplecticMatrix{T}}, Q::SymplecticMatrix) where {T}
    return SymplecticMatrix(convert(T, Q.λ))
end

function Base.show(io::IO, Q::SymplecticMatrix)
    s = "$(Q.λ)"
    if occursin(r"\w+\s*[\+\-]\s*\w+", s)
        s = "($s)"
    end
    return print(io, typeof(Q), "(): $(s)*[0 I; -I 0]")
end

(Base.:-)(Q::SymplecticMatrix) = SymplecticMatrix(-Q.λ)

(Base.:*)(x::Number, Q::SymplecticMatrix) = SymplecticMatrix(x * Q.λ)
(Base.:*)(Q::SymplecticMatrix, x::Number) = SymplecticMatrix(x * Q.λ)
function (Base.:*)(Q1::SymplecticMatrix, Q2::SymplecticMatrix)
    return LinearAlgebra.UniformScaling(-Q1.λ * Q2.λ)
end

Base.transpose(Q::SymplecticMatrix) = -Q
Base.adjoint(Q::SymplecticMatrix) = -Q
Base.inv(Q::SymplecticMatrix) = SymplecticMatrix(-(1 / Q.λ))

(Base.:+)(Q1::SymplecticMatrix, Q2::SymplecticMatrix) = SymplecticMatrix(Q1.λ + Q2.λ)
(Base.:-)(Q1::SymplecticMatrix, Q2::SymplecticMatrix) = SymplecticMatrix(Q1.λ - Q2.λ)

(Base.:+)(Q::SymplecticMatrix, p::AbstractMatrix) = p + Q
function (Base.:+)(p::AbstractMatrix, Q::SymplecticMatrix)
    # When we are adding, the Matrices must match in size:
    n, _ = get_even_dims(p; square=true)

    # Allocate new memory:
    TS = Base._return_type(+, Tuple{eltype(p),eltype(Q)})
    out = copyto!(similar(p, TS), p)

    # Add Q.λ multiples of the UniformScaling to the lower left and upper right blocks of p:
    λ_Id = LinearAlgebra.UniformScaling(Q.λ)

    out[1:n, (n + 1):(2n)] += λ_Id
    out[(n + 1):(2n), 1:n] -= λ_Id
    return out
end

# Binary minus:
(Base.:-)(Q::SymplecticMatrix, p::AbstractMatrix) = Q + (-p)
(Base.:-)(p::AbstractMatrix, Q::SymplecticMatrix) = p + (-Q)

function (Base.:*)(p::AbstractMatrix, Q::SymplecticMatrix)
    _, k = Manifolds.get_even_dims(p)

    # Allocate new memory:
    TS = typeof(one(eltype(p)) + one(eltype(Q)))
    pQ = similar(p, TS)

    # Perform right mulitply by λ*Q:
    mul!((@inbounds view(pQ, :, 1:k)), -Q.λ, @inbounds view(p, :, (k + 1):lastindex(p, 2)))
    mul!((@inbounds view(pQ, :, (k + 1):lastindex(pQ, 2))), Q.λ, @inbounds view(p, :, 1:k))
    return pQ
end

function (Base.:*)(Q::SymplecticMatrix, p::AbstractMatrix)
    n, _ = Manifolds.get_even_dims(p)

    # Allocate new memory:
    TS = typeof(one(eltype(p)) + one(eltype(Q)))
    Qp = similar(p, TS)

    # Perform left mulitply by λ*Q:
    mul!((@inbounds view(Qp, 1:n, :)), Q.λ, @inbounds view(p, (n + 1):lastindex(p, 1), :))
    mul!((@inbounds view(Qp, (n + 1):lastindex(Qp, 1), :)), -Q.λ, @inbounds view(p, 1:n, :))

    return Qp
end

function LinearAlgebra.lmul!(Q::SymplecticMatrix, p::AbstractMatrix)
    # Perform left multiplication by a symplectic matrix,
    # overwriting the matrix p in place:
    n, _ = get_even_dims(p)

    half_row_p = copy(@inbounds view(p, 1:n, :))

    mul!((@inbounds view(p, 1:n, :)), Q.λ,
          @inbounds view(p, (n + 1):lastindex(p, 1), :))

    mul!((@inbounds view(p, (n+1):lastindex(p, 1), :)), -Q.λ,
          @inbounds view(half_row_p, :, :))
    return p
end

function LinearAlgebra.rmul!(p::AbstractMatrix, Q::SymplecticMatrix)
    # Perform right multiplication by a symplectic matrix,
    # overwriting the matrix p in place:
    _, k = get_even_dims(p)

    half_col_p = copy(@inbounds view(p, :, 1:k))

    mul!((@inbounds view(p, :, 1:k)), -Q.λ,
          @inbounds view(p, :, (k + 1):lastindex(p, 2)))

    mul!((@inbounds view(p, :, (k+1):lastindex(p, 2))), Q.λ,
          @inbounds view(half_col_p, :, :))

    return p
end

function LinearAlgebra.mul!(A::AbstractMatrix, p::AbstractMatrix, Q::SymplecticMatrix)
    _, k = get_even_dims(p)
    # Perform right mulitply by λ*Q:
    mul!((@inbounds view(A, :, 1:k)), -Q.λ, @inbounds view(p, :, (k + 1):lastindex(p, 2)))
    mul!((@inbounds view(A, :, (k + 1):lastindex(A, 2))), Q.λ, @inbounds view(p, :, 1:k))
    return A
end

function LinearAlgebra.mul!(A::AbstractMatrix, Q::SymplecticMatrix, p::AbstractMatrix)
    n, _ = get_even_dims(p)
    # Perform left mulitply by λ*Q:
    mul!((@inbounds view(A, 1:n, :)), Q.λ, @inbounds view(p, (n + 1):lastindex(p, 1), :))
    mul!((@inbounds view(A, (n + 1):lastindex(A, 1), :)), -Q.λ, @inbounds view(p, 1:n, :))
    return A
end

function add_scaled_I!(A::AbstractMatrix, λ::Number)
    LinearAlgebra.checksquare(A)
    @inbounds for i in axes(A, 1)
        A[i, i] += λ
    end
    return A
end

@doc raw"""
    inv(M::Symplectic{n, ℝ}, A) where {n, ℝ}

Compute the symplectic inverse ``A^+`` of matrix ``A ∈ ℝ^{2n × 2n}``. Given a matrix
````math
A ∈ ℝ^{2n × 2n},\quad
A =
\begin{bmatrix}
A_{1,1} & A_{1,2} \\
A_{2,1} & A_{2, 2}
\end{bmatrix}
````
the symplectic inverse is defined as:
````math
A^{+} := Q_{2n}^T A^T Q_{2n},
````
where
````math
Q_{2n} =
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n
\end{bmatrix}.
````
The symplectic inverse of A can be expressed explicitly as:
````math
A^{+} =
\begin{bmatrix}
  A_{2, 2}^T & -A_{1, 2}^T \\[1.2mm]
 -A_{2, 1}^T &  A_{1, 1}^T
\end{bmatrix}.
````
"""
function Base.inv(::Symplectic{n,ℝ}, A) where {n}
    Ai = similar(A)
    checkbounds(A, 1:(2n), 1:(2n))
    @inbounds for i in 1:n, j in 1:n
        Ai[i, j] = A[j + n, i + n]
    end
    @inbounds for i in 1:n, j in 1:n
        Ai[i + n, j] = -A[j + n, i]
    end
    @inbounds for i in 1:n, j in 1:n
        Ai[i, j + n] = -A[j, i + n]
    end
    @inbounds for i in 1:n, j in 1:n
        Ai[i + n, j + n] = A[j, i]
    end
    return Ai
end

function inv!(::Symplectic{n,ℝ}, A) where {n}
    checkbounds(A, 1:(2n), 1:(2n))
    @inbounds for i in 1:n, j in 1:n
        tmp = A[i, j]
        A[i, j] = A[j + n, i + n]
        A[j + n, i + n] = tmp
    end
    @inbounds for i in 1:n, j in i:n
        if i == j
            A[i, j + n] = -A[i, j + n]
        else
            tmp = A[i, j + n]
            A[i, j + n] = -A[j, i + n]
            A[j, i + n] = -tmp
        end
    end
    @inbounds for i in 1:n, j in i:n
        if i == j
            A[i + n, j] = -A[i + n, j]
        else
            tmp = A[i + n, j]
            A[i + n, j] = -A[j + n, i]
            A[j + n, i] = -tmp
        end
    end
    return A
end

function symplectic_inverse_times(M::Symplectic{n}, p, q) where {n}
    A = similar(p)
    return symplectic_inverse_times!(M, A, p, q)
end
function symplectic_inverse_times!(::Symplectic{n}, A, p, q) where {n}
    # we write p = [p1 p2; p3 p4] (and q, too), then
    p1 = @view(p[1:n, 1:n])
    p2 = @view(p[1:n, (n + 1):(2n)])
    p3 = @view(p[(n + 1):(2n), 1:n])
    p4 = @view(p[(n + 1):(2n), (n + 1):(2n)])
    q1 = @view(q[1:n, 1:n])
    q2 = @view(q[1:n, (n + 1):(2n)])
    q3 = @view(q[(n + 1):(2n), 1:n])
    q4 = @view(q[(n + 1):(2n), (n + 1):(2n)])
    A1 = @view(A[1:n, 1:n])
    A2 = @view(A[1:n, (n + 1):(2n)])
    A3 = @view(A[(n + 1):(2n), 1:n])
    A4 = @view(A[(n + 1):(2n), (n + 1):(2n)])
    mul!(A1, p4', q1) # A1 = p4'q1
    mul!(A1, p2', q3, -1, 1) # A1 -= p2'p3
    mul!(A2, p4', q2) # A2 = p4'q2
    mul!(A2, p2', q4, -1, 1) #A2 -= p2'q4
    mul!(A3, p1', q3) #A3 = p1'q3
    mul!(A3, p3', q1, -1, 1) # A3 -= p3'q1
    mul!(A4, p1', q4) # A4 = p1'q4
    mul!(A4, p3', q2, -1, 1) #A4 -= p3'q2
    return A
end

function rand_hamiltonian(::Symplectic{n}; final_norm=1) where {n}
    A = randn(n, n)
    B = randn(n, n)
    C = randn(n, n)
    B = (1 / 2) .* (B .+ B')
    C = (1 / 2) .* (C .+ C')
    Ω = [A B; C -A']
    return final_norm * Ω / norm(Ω, 2)
end

@doc raw"""
    grad_euclidean_to_manifold(M::Symplectic{n}, p, ∇f_euc)
    grad_euclidean_to_manifold!(M::Symplectic{n}, ∇f_man, p, ∇f_euc)

Compute the transformation of the euclidean gradient of a scalar function `f` onto the
tangent space of the point ``p \in \operatorname{Sp}(ℝ, 2n)``[^FioriSimone2011].
The transformation is found by requireing that the gradient element in the tangent
space solves the metric compatibility for the Riemannian default_metric_dispatch
along with the defining equation for a tangent
vector ``X \in T_pSn(ℝ)``at a point ``p \in Sn(ℝ)``.

First we change the representation of the gradient from the Euclidean metric to the RealSymplecticMetric at p,
and then we project the result onto the tangent space ``T_p\operatorname{Sp}(2n, ℝ)`` at p.

[^FioriSimone2011]:
    > Simone Fiori:
    > Solving minimal-distance problems over the manifold of real-symplectic matrices,
    > SIAM Journal on Matrix Analysis and Applications 32(3), pp. 938-968, 2011.
    > doi [10.1137/100817115](https://doi.org/10.1137/100817115).
"""
function grad_euclidean_to_manifold(M::Symplectic{n}, p, ∇f_euc) where {n}
    return grad_euclidean_to_manifold!(M, similar(∇f_euc), p, ∇f_euc)
end

function grad_euclidean_to_manifold!(M::Symplectic{n}, ∇f_man, p, ∇f_euc) where {n}
    make_metric_compatible!(M, ∇f_man, EuclideanMetric(), p, ∇f_euc)
    return project_riemannian!(M, ∇f_man, p, ∇f_man)
end

# Overwrite gradient functions for the Symplectic case:
# Need to first change representer of ``∇f_euc`` to the Symplectic manifold,
# then project onto the correct tangent space.
@doc raw"""
    gradient(M::Symplectic, f, p, backend::RiemannianProjectionBackend)

Specialize the `gradient` computation of scalar functions
``f \colon \operatorname{Sp} \rightarrow \mathbb{R}``
to use the Riemannian metric projection after a metric compatibility mapping.
"""
function gradient(M::Symplectic, f, p, backend::RiemannianProjectionBackend)
    return gradient!(M, f, similar(p), p, backend)
end

function gradient!(M::Symplectic, f, X, p, backend::RiemannianProjectionBackend)
    _gradient!(f, X, p, backend.diff_backend)
    make_metric_compatible!(M, X, EuclideanMetric(), p, X)
    return project_riemannian!(M, X, p, X)
end

@doc raw"""
    make_metric_compatible(::Symplectic, ::EuclideanMetric, p, X)
    make_metric_compatible!(::Symplectic, Y, ::EuclideanMetric, p, X)

Change the representation of an arbitrary element ``χ ∈ \mathbb{R}^{2n \times 2n}``
by a mapping
````math
    c_p : \mathbb{R}^{2n \times 2n} \rightarrow \mathbb{R}^{2n \times 2n},
    \quad c_p(χ) = pp^T χ.
````
The mapping is defined such that the metric compatibility condition
````math
    g_p(c_p(χ), η) = ⟨χ, η⟩^{\text{Euc}} \;∀\; η ∈ T_p\operatorname{Sp}(2n, ℝ)
````
holds, where ``g`` is the Riemannian metric [`RealSymplecticMetric`](@ref).
"""
function make_metric_compatible(M::Symplectic, EucMet::EuclideanMetric, p, X)
    return make_metric_compatible!(M, similar(X), EucMet, p, X)
end

function make_metric_compatible!(::Symplectic, Y, ::EuclideanMetric, p, X)
    Y .= p * p' * X
    return Y
end


@doc raw"""
    change_representer!(::Symplectic, Y, ::EuclideanMetric, p, X)

Compute the representation of a tangent vector ``χ ∈ T_p\operatorname{Sp}(2n, ℝ)`` s.t.
````math
    g_p(c_p(χ), η) = ⟨χ, η⟩^{\text{Euc}} \;∀\; η ∈ T_p\operatorname{Sp}(2n, ℝ).
````
with the conversion function
````math
    c_p : T_p\operatorname{Sp}(2n, ℝ) \rightarrow T_p\operatorname{Sp}(2n, ℝ), \quad
    c_p(η) = \frac{1}{2} pp^T η + \frac{1}{2} pQ η^T pQ.
````

Each of the terms ``c_p^1(η) = p p^T η`` and ``c_p^2(η) = pQ η^T pQ`` from the
above definition of ``c_p(η)`` are themselves metric compatible in the sense that
````math
    c_p^i : T_p\operatorname{Sp}(2n, ℝ) \rightarrow \mathbb{R}^{2n \times 2n}\quad
    g_p^i(c_p(χ), η) = ⟨χ, η⟩^{\text{Euc}} \;∀\; η ∈ T_p\operatorname{Sp}(2n, ℝ),
````
for ``i \in {1, 2}``. However the range of each function alone is not confined to
``T_p\operatorname{Sp}(2n, ℝ)``, but the convex combination
````math
    c_p(η) = \frac{1}{2}c_p^1(η) + \frac{1}{2}c_p^2(η)
````
does have the correct range ``T_p\operatorname{Sp}(2n, ℝ)``.
"""
function change_representer!(::Symplectic, Y, ::EuclideanMetric, p, X)
    Q = SymplecticMatrix(p, X)
    Y .= (1 / 2) .* p * (p' * X .+ Q * X' * p * Q)
    return Y
end

@doc raw"""
    project_riemannian!(M::Symplectic{n, ℝ}, Y, p, X) where {n}

Compute the projection of ``X ∈ R^{2n × 2n}`` onto ``T_p\operatorname{Sp}(2n, ℝ)`` w.r.t.
the Riemannian metric ``g`` [`RealSymplecticMetric`](@ref).

The closed form projection mapping is given by [^Gao2021riemannian]
````math
    \operatorname{P}^{T_p\operatorname{Sp}(2n)}_{g_p}(X) = pQ\operatorname{sym}(p^TQ^TX),
````
where ``\operatorname{sym}(A) = \frac{1}{2}(A + A')``.
"""
function project_riemannian!(::Symplectic{n,ℝ}, Y, p, X) where {n}
    Q = SymplecticMatrix(p, X)

    pT_QT_X = p' * Q' * X
    symmetrized_pT_QT_X = (1 / 2) .* (pT_QT_X + pT_QT_X')

    Y[:, :] = p * Q * (symmetrized_pT_QT_X)
    return Y
end

@doc raw"""
    project_riemannian_normal!(M::Symplectic{n, ℝ}, Y, p, X)

Project onto the normal of the tangent space ``(T_p\operatorname{Sp}(2n))^{\perp_g}`` at
a point ``p ∈ \operatorname{Sp}(2n)``,
relative to the riemannian metric ``g`` [`RealSymplecticMetric`](@ref).
That is,
````math
(T_p\operatorname{Sp}(2n))^{\perp_g} = \{Y \in \mathbb{R}^{2n \times 2n} :
                        g_p(Y, X) = 0 \;\forall\; X \in T_p\operatorname{Sp}(2n)\}.
````
The closed form projection operator onto the normal space is given by [^Gao2021riemannian]
````math
\operatorname{P}^{(T_p\operatorname{Sp}(2n))\perp}_{g_p}(X) = pQ\operatorname{skew}(p^TQ^TX),
````
where ``\operatorname{skew}(A) = \frac{1}{2}(A - A')``.

[^Gao2021riemannian]:
    > Gao, Bin and Son, Nguyen Thanh and Absil, P-A and Stykel, Tatjana:
    > Riemannian optimization on the symplectic Stiefel manifold,
    > SIAM Journal on Optimization 31(2), pp. 1546-1575, 2021.
    > doi [10.1137/20M1348522](https://doi.org/10.1137/20M1348522)
"""
function project_riemannian_normal!(::Symplectic{n,ℝ}, Y, p, X) where {n}
    Q = SymplecticMatrix(p, X)

    pT_QT_X = p' * Q' * X
    skew_pT_QT_X = (1 / 2) .* (pT_QT_X .- pT_QT_X')

    Y[:, :] = p * Q * skew_pT_QT_X
    return Y
end

@doc raw"""
    retract(::Symplectic, p, X, ::CayleyRetraction)

Compute the Cayley retraction on ``p ∈ \operatorname{Sp}(2n, ℝ)`` in
the direction of tangent vector ``X ∈ T_p\operatorname{Sp}(2n, ℝ)``,
as defined in by Birtea et al in proposition 2 [^birtea2020optimization].

Using the symplectic inverse of a matrix ``A \in ℝ^{2n \times 2n}``,
``
A^{+} := Q_{2n}^T A^T Q_{2n}
``
where
````math
Q_{2n} =
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n
\end{bmatrix},
````
the retraction
``\mathcal{R}\colon T\operatorname{Sp}(2n) \rightarrow \operatorname{Sp}(2n)``
is defined pointwise as
````math
\begin{align*}
\mathcal{R}_p(X) &= p \operatorname{cay}\left(\frac{1}{2}p^{+}X\right), \\
                 &= p \operatorname{exp}_{1/1}(p^{+}X), \\
                 &= p (2I - p^{+}X)^{-1}(2I + p^{+}X).
\end{align*}
````
Here
``\operatorname{exp}_{1/1}(z) = (2 - z)^{-1}(2 + z)``
denotes the Padé (1, 1) approximation to ``\operatorname{exp}(z)``.

[^birtea2020optimization]:
    > Birtea, Petre and Ca{\c{s}}u, Ioan and Com{\u{a}}nescu, Dan:
    > Optimization on the real symplectic group,
    > Monatshefte f{\"u}r Mathematik, Springer
    > doi [10.1007/s00605-020-01369-9](https://doi.org/10.1007/s00605-020-01369-9)
"""
function retract!(M::Symplectic, q, p, X, ::CayleyRetraction)
    p_star_X = symplectic_inverse_times(M, p, X)

    ldiv!(lu!(2*I - p_star_X), add_scaled_I!(p_star_X, 2.0))
    q .= p * p_star_X
    return q
end

ManifoldsBase.default_retraction_method(::Symplectic) = CayleyRetraction()

struct CayleyInverseRetraction <: AbstractInverseRetractionMethod end

@doc raw"""
    inverse_retract!(M::Symplectic, X, p, q, ::CayleyInverseRetraction)

Compute the Cayley Inverse Retraction as in proposition 5.3 of
Bendorkat & Zimmermann[^Bendokat2021].

First, recall the definition the standard symplectic matrix
````math
Q =
\begin{bmatrix}
 0    & I \\
-I  & 0
\end{bmatrix}
````
as well as the symplectic inverse of a matrix ``A``, ``A^{+} = Q^T A^T Q``.

For ``p, q ∈ \operatorname{Sp}(2n, ℝ)`` then, we can then define the
inverse cayley retraction as long as the following matrices exist.
````math
    U = (I + p^+ q)^{-1}, \quad V = (I + q^+ p)^{-1}.
````

If that is the case, the inverse cayley retration at ``p`` applied to ``q`` is
````math
\mathcal{L}_p^{\operatorname{Sp}}(q) = 2p\bigl(V - U\bigr) + 2\bigl((p + q)U - p\bigr)
                                        ∈ T_p\operatorname{Sp}(2n).
````

[^Bendokat2021]:
    > Bendokat, Thomas and Zimmermann, Ralf:
	> The real symplectic Stiefel and Grassmann manifolds: metrics, geodesics and applications
	> arXiv preprint arXiv:2108.12447, 2021 (https://arxiv.org/abs/2108.12447)
"""
function inverse_retract!(M::Symplectic, X, p, q, ::CayleyInverseRetraction)
    U_inv = lu!(add_scaled_I!(symplectic_inverse_times(M, p, q), 1))
    V_inv = lu!(add_scaled_I!(symplectic_inverse_times(M, q, p), 1))

    X .= 2 .* ((p / V_inv .- p / U_inv) .+ ((p + q) / U_inv) .- p)
    return X
end
