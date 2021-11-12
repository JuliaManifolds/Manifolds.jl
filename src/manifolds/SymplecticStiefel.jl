@doc raw"""
Working memory for SymplecticStiefel Exp (Retraction).
"""
struct SymplecticRetraction{T <:AbstractMatrix}
    m_2k_2k::T
    m_2n_2k::T
    m_2k_2n::T
end

@doc raw"""
Working memory for SymplecticStiefel Inner product.
"""
struct SymplecticInner{T<:AbstractMatrix}
    m_2k_2k::T
    m_2n_2k::T
    m_2k_2n::T
end

@doc raw"""
Working memory for SymplecticStiefel Gradient Conversion.
"""
struct SymplecticGrad{T<:AbstractMatrix}
    m_2k_2k::T
    m_2n_2k::T
    m_2k_2n::T
end

#=
# TODO: Store all matrices in a single struct like this, and store a single
#       instance of this struct in the SymplecticStiefel manifold. which
#       only gets populated with memory if any functions needing it are called.
struct SymplecticStorage{T<:AbstractMatrix}
    m_2k_2k::T
    m_2n_2k::T
    m_2k_2n::T
end
=#

@doc raw"""
The Symplectic Stiefel manifold. Each element represent a Symplectic Subspace of ``ℝ^{2n × 2k}``.
"""
mutable struct SymplecticStiefel{n, k, 𝔽, T} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}
    # May have to specify type of 'points' and eltype {S, T <: AbstractMatrix{<:S}}
    # To avoid type instability.
    symplectic_retraction::SymplecticRetraction{T}
    symplectic_inner::SymplecticInner{T}
    symplectic_grad::SymplecticGrad{T}

    function SymplecticStiefel{n, k, 𝔽, T}() where {n, k, 𝔽, T}
        return new{n, k, 𝔽, T}()
    end
end

function SymplecticInner(M::SymplecticStiefel{n, k}, p::T) where {n, k, S, T<:AbstractMatrix{<:S}}
    isdefined(M, :symplectic_inner) && return M.symplectic_inner
    TS = eltype(p)
    M.symplectic_inner = SymplecticInner{T}(zeros(TS, 2k, 2k),
                                            zeros(TS, 2n, 2k),
                                            zeros(TS, 2k, 2n))
    return M.symplectic_inner
end

function (F::SymplecticInner)(::SymplecticStiefel{n, k}, p, X, Y) where {n, k}
    # Functor edition:
    # Need to do the same to this function, reduce allocations as much as possible:
    # Do not use any (2n × 2n) matrices.

    # Avoid constructing the (2n, 2n) matrix.
    Q = SymplecticMatrix(p, X, Y)

    # Perform LU-factorization before multiplication:
    F.m_2k_2k .= p' * p
    p_Tp = lu!(F.m_2k_2k)

    # Use the p_Tp-memory for the requred solves:
    # Y / p_Tp -> M.m_2n_2k / p_Tp -> F.m_2n_2k
    F.m_2n_2k .= Y
    rdiv!(F.m_2n_2k, p_Tp)

    # p_Tp \ (p') -> M.m_2k_2n:
    ldiv!(F.m_2k_2n, p_Tp, p')

    ##########################################
    #### F.m_2k_2k now available for use. ####
    ##########################################

    # Compute first term: tr(X^T * Y * (p^T * p)^{-1})
    # X' * (Y / p_Tp) -> F.m_2k_2k
    mul!(F.m_2k_2k, X', F.m_2n_2k)

    first_term = tr(F.m_2k_2k)

    # Do not construct the inner matrix ℝ^{2n × 2n}:
    # (I - (1/2) * Q' * p * (p_Tp \ (p')) * Q)
    # Instead make:
    # X'*Q'*p*(p' * p)^{-1} * p' * Q * Y *(p' * p)^{-1} ∈ ℝ^{2k × 2k}.

    # Q * (Y / p_Tp) -> F.m_2n_2k
    lmul!(Q, F.m_2n_2k)

    # (p_Tp \ (p')) * (Q * (Y / p_Tp)) -> F.m_2k_2k
    mul!(F.m_2k_2k, F.m_2k_2n, F.m_2n_2k)

    ########################################################
    #### M.m_2n_2k and M.m_2k_2n now available for use. ####
    ########################################################

    # Copy p -> M.m_2n_2k:
    F.m_2n_2k .= p

    # Q' *p -> F.m_2n_2k
    lmul!(Q', F.m_2n_2k)

    # DANGER: May introduce Allocations, but should just overwrite.
    # Q'*p * (p_Tp \ (p')) * (Q * (Y / p_Tp)) -> F.m_2n_2k
    F.m_2n_2k .= F.m_2n_2k * F.m_2k_2k

    ##########################################
    #### F.m_2k_2k now available for use. ####
    ##########################################

    # X' * Q' *p * (p_Tp \ (p')) * (Q * (Y / p_Tp)) -> F.m_2k_2k
    mul!(F.m_2k_2k, X', F.m_2n_2k)

    second_term = tr(F.m_2k_2k)
    return first_term - (1/2)*second_term
end

function SymplecticRetraction(M::SymplecticStiefel{n, k}, p::T) where {n, k, S, T<:AbstractMatrix{<:S}}
    isdefined(M, :symplectic_retraction) && return M.symplectic_retraction
    TS = eltype(p)
    M.symplectic_retraction = SymplecticRetraction{T}(zeros(TS, 2k, 2k),
                                                      zeros(TS, 2n, 2k),
                                                      zeros(TS, 2k, 2n))
    return M.symplectic_retraction
end

function (F::SymplecticRetraction)(M::SymplecticStiefel{n, k}, q, p, X) where {n, k}
    # The new one!
    # About half to a third as much allocation done here:
    # Believe this version of inv!(M, q, p) is the fastets:
    F.m_2k_2k .= inv!(M, F.m_2k_2n, p) * X
    F.m_2n_2k .= X .- p*F.m_2k_2k

    # Make the 'dividend' and store into 'M.retr_A' using
    # the five-argument version of 'mul!()':
    # (-0.5*retr_A + 0.25*inv(M, M.retr_H)*M.retr_H -> retr_A.
    mul!(F.m_2k_2k, inv!(M, F.m_2k_2n, F.m_2n_2k), F.m_2n_2k, 0.25, -0.5)
    add_scaled_I!(F.m_2k_2k, 1.0)

    # lu! Should not allocate new memory. Overwrites F.m_2k_2k
    factored_dividend = lu!(F.m_2k_2k)

    # Add two copies of p to F.m_2n_2k:
    F.m_2n_2k .+= 2*p

    q .= ((-1.0) .* p) .+ rdiv!(F.m_2n_2k, factored_dividend)
    return q
end


function SymplecticGrad(M::SymplecticStiefel{n, k}, p::T) where {n, k, S, T<:AbstractMatrix{<:S}}
    isdefined(M, :symplectic_grad) && return M.symplectic_grad
    TS = eltype(p)
    M.symplectic_grad = SymplecticGrad{T}(zeros(TS, 2k, 2k),
                                          zeros(TS, 2n, 2k),
                                          zeros(TS, 2k, 2n))
    return M.symplectic_grad
end


function (F::SymplecticGrad)(M::SymplecticStiefel, ∇f_man, p, ∇f_euc)
    Q = SymplecticMatrix(p, ∇f_euc)

    # Avoid constructing 2n × 2n matrices.
    # grad_F = (∇F * p' * p) + (Q * p * ∇F' * Q * p)

    ### First term:
    # p' * p -> F.m_2k_2k
    mul!(F.m_2k_2k, p', p)
    # ∇F * (p' * p) -> ∇f_man
    mul!(∇f_man, ∇f_euc, F.m_2k_2k)

    ### Second term:
    # Q * p -> F.m_2n_2k
    F.m_2n_2k .= p
    lmul!(Q, F.m_2n_2k)

    # ∇F' * (Q * p) -> F.m_2k_2k
    mul!(F.m_2k_2k, ∇f_euc', F.m_2n_2k)

    # p * (∇F' * Q * p) -> F.m_2n_2k
    mul!(F.m_2n_2k, p, F.m_2k_2k)
    # Q * (p * ∇F' * Q * p) -> F.m_2n_2k
    lmul!(Q, F.m_2n_2k)

    # ∇f_man .= ∇f_euc * p' * p .+ Q * p * (∇f_euc)' * Q * p
    ∇f_man .+= F.m_2n_2k
    return ∇f_man
end


@doc raw"""
    You are given a manifold of embedding dimension two_n × two_p.
    # Tried to type the fields being stored in SymplecticStiefel as well.
"""
function SymplecticStiefel(two_n::Int, two_k::Int, field::AbstractNumbers=ℝ,
                           T::Type=Matrix{Float64})
    SymplecticStiefel{div(two_n, 2), div(two_k, 2), field, T}()
end

function SymplecticStiefel(two_n::Int, two_k::Int, ::T) where {T <: AbstractMatrix}
    SymplecticStiefel{div(two_n, 2), div(two_k, 2), ℝ, T}()
end

Base.show(io::IO, ::SymplecticStiefel{n, k}) where {n, k} = print(io, "SymplecticStiefel{$(2n), $(2k)}()")

decorated_manifold(::SymplecticStiefel{n, k, ℝ}) where {n, k} = Euclidean(2n, 2k; field=ℝ)
ManifoldsBase.default_retraction_method(::SymplecticStiefel) = CayleyRetraction()

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
function inner_old(::SymplecticStiefel{n, k}, p, X, Y) where {n, k}
    Q = SymplecticMatrix(p, X, Y)
    # Procompute lu(p'p) since we solve a^{-1}* 3 times
    a = lu(p'*p) # note that p'p is symmetric so is its inverse c=a^{-1}
    b = Q'*p
    # we split the original trace into two one with I->(X'Yc)
    # and the other with 1/2 X'b c b' Y c
    # a) we permute X' and Y c to c^TY^TX = a\(Y'X) (avoids a large interims matrix)
    # b) we permute Y c up front, the center term is symmetric, so we get cY'b c b' X
    # and (b'X) again avoids a large interims matrix, so does Y'b.
    return return tr(a\(Y'*X)) - 1//2 * tr( a\( (Y' * b) * (a \ (b' * X)) ) )
end

function inner(M::SymplecticStiefel{n, k}, p, X, Y) where {n, k}
    return SymplecticInner(M, p)(M, p, X, Y)
end

function inner_deprecated(M::SymplecticStiefel{n, k}, p, X, Y) where {n, k}
    # The new one!
    # Need to do the same to this function, reduce allocations as much as possible:

    # Avoid constructing the (2n, 2n) matrix.
    Q = SymplecticMatrix(p, X, Y)

    # Perform LU-factorization before multiplication:
    M.m_2k_2k .= p' * p
    p_Tp = lu!(M.m_2k_2k)

    # p_Tp \ (p') -> M.m_2k_2n:
    ldiv!(M.m_2k_2n, p_Tp, p')

    # Construct inner matrix:
    # (I - (1/2) * Q' * p * (p_Tp \ (p')) * Q)

    # Copy p -> M.m_2n_2k:
    M.m_2n_2k .= p
    # (-1/2)*Q'*p -> M.m_2n_2k
    lmul!((-1/2) * Q', M.m_2n_2k)

    # (p_Tp \ (p')) * Q -> M.m_2k_2n
    rmul!(M.m_2k_2n, Q)

    mul!(M.m_2n_2n, M.m_2n_2k, M.m_2k_2n)
    add_scaled_I!(M.m_2n_2n, 1.0)

    ########################################################
    #### M.m_2n_2k and M.m_2k_2n now available for use. ####
    ########################################################

    # println("Inner matrix:")
    # display(M.m_2n_2n)

    # Use the p_Tp-memory for the requred solve:
    # Y / p_Tp -> M.m_2n_2k / p_Tp -> M.m_2n_2k
    M.m_2n_2k .= Y
    rdiv!(M.m_2n_2k, p_Tp)

    ##########################################
    #### M.m_2k_2k now available for use. ####
    ##########################################

    # X' * (I - (1/2) * Q' * p * (p_Tp \ (p')) * Q) -> M.m_2k_2n
    mul!(M.m_2k_2n, X', M.m_2n_2n)

    # Lastly: X' * (I - (1/2) * Q' * p * (p_Tp \ (p')) * Q) * (Y / p_Tp) -> M.m_2k_2k
    # Compute tr(A'*B) with for-loops? Saves this computation.

    mul!(M.m_2k_2k, M.m_2k_2n, M.m_2n_2k)
    return tr(M.m_2k_2k)
end

function Base.inv(M::SymplecticStiefel{n, k}, p) where {n, k}
    q = similar(p')
    inv!(M, q, p)
end

function inv!(::SymplecticStiefel{n,k}, q, p) where {n, k}
    checkbounds(q, 1:2k, 1:2n)
    checkbounds(p, 1:2n, 1:2k)
    @inbounds for i in 1:k, j in 1:n
        q[i, j] = p[j+n, i+k]
    end
    @inbounds for i in 1:k, j in 1:n
        q[i, j+n] = -p[j, i+k]
    end
    @inbounds for i in 1:k, j in 1:n
        q[i+k, j] = -p[j+n, i]
    end
    @inbounds for i in 1:k, j in 1:n
        q[i+k, j+n] = p[j, i]
    end
    return q
end

function old_inv!(::SymplecticStiefel{n,k}, q, p) where {n, k}
    q[1:k, 1:n] .= p[(n+1):2n, (k+1):2k]'
    q[1:k, (n+1):2n] .= -p[1:n, (k+1):2k]'
    q[(k+1):2k, 1:n] .= -p[(n+1):2n, 1:k]'
    q[(k+1):2k, (n+1):2n] .= p[1:n, 1:k]'
    return q
end

@doc raw"""
    change_representer!(::SymplecticStiefel{n, k}, Y, ::EuclideanMetric, p, X)

Calculated myself, but tailored to the right-invariant inner product of Bendokat-Zimmermann
"""
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
    canonical_projection(M, rand(Symplectic(2n)))
end

@doc raw"""
    rand(::SymplecticStiefel{n, k}, p)

As based on the parametrization of the tangent space ``T_p\operatorname{SpSt}(n, k)`` found in Proposition 3.2
of Benodkat-Zimmermann. There they express the tangent space as ``X = pΩ + p^sB``, where ``Ω^+ = -Ω`` is Hamiltonian.
The notation ``p^s`` means the symplectic complement of ``p`` s.t. ``p^{+}p^{s} = 0``, and ``B ∈ ℝ^{2(n-k) × 2k}.
"""
Base.rand(::SymplecticStiefel{n, k}, p) where {n, k} = begin
    Ω = rand_hamiltonian(Symplectic(2k))
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
    return SymplecticRetraction(M, p)(M, q, p, X)
end

# compute p^+q (which is 2kx2k) in place of A
function symplectic_inverse_times(M::SymplecticStiefel{n,k}, p, q) where {n,k}
    A = similar(p, (2k,2k))
    return symplectic_inverse_times!(M, A, p, q)
end
function symplectic_inverse_times!(::SymplecticStiefel{n,k}, A, p, q) where {n,k}
    # we write p = [p1 p2; p3 p4] (and q, too), then
    p1 = @view(p[1:n,1:k])
    p2 = @view(p[1:n,(k+1):2k])
    p3 = @view(p[(n+1):2n,1:k])
    p4 = @view(p[(n+1):2n,(k+1):2k])
    q1 = @view(q[1:n,1:k])
    q2 = @view(q[1:n,(k+1):2k])
    q3 = @view(q[(n+1):2n,1:k])
    q4 = @view(q[(n+1):2n,(k+1):2k])
    A1 = @view(A[1:k,1:k])
    A2 = @view(A[1:k,(k+1):2k])
    A3 = @view(A[(k+1):2k,1:k])
    A4 = @view(A[(k+1):2k,(k+1):2k])
    mul!(A1, p4', q1) # A1 = p4'q1
    mul!(A1, p2', q3, -1, 1) # A1 -= p2'p3
    mul!(A2, p4', q2)
    mul!(A2, p2', q4, -1, 1)
    mul!(A3, p1', q3)
    mul!(A3, p3', q1, -1, 1)
    mul!(A4, p1', q4)
    mul!(A4, p3', q2, -1, 1)
    return A
end

function retract_old!(M::SymplecticStiefel{n, k}, q, p, X, ::CayleyRetraction) where {n, k}
    # Define intermediate matrices for later use:
    #A = inv(M, p) * X # 2k x 2k - writing this out explicitly, since this allocates a 2kx2n matrix.
    A = symplectic_inverse_times(M, p, X)
    q .= X .- p*A # H in BZ21
    A .= -A./2 .+ symplectic_inverse_times(M, q, q)./4 #-A/2 + H^+H/4
    q .= q .+ 2 .* p
    q .= -p .+ q / lu((I + A))
    return q
end

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
    return ∇f_euc * (p' * p)  .+ Q * p * (∇f_euc' * Q * p)
end

function grad_euclidian_to_manifold_old!(::SymplecticStiefel, ∇f_man, p, ∇f_euc)
    Q = SymplecticMatrix(p, ∇f_euc)
    ∇f_man .= (∇f_euc * p' .+ Q * p * (∇f_euc)' * Q) * p
    return ∇f_man
end

function grad_euclidian_to_manifold!(M::SymplecticStiefel, ∇f_man, p, ∇f_euc)
    SymplecticGrad(M, p)(M, ∇f_man, p, ∇f_euc)
end
