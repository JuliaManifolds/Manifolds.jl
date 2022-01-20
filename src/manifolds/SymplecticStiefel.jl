@doc raw"""
    SymplecticStiefel{n, k, ℝ} <: AbstractEmbeddedManifold{ℝ, DefaultIsometricEmbeddingType}

Over the real number field ℝ the elements of the symplectic Stiefel manifold
are all $2n × 2k, \; k \leq n$ matrices satisfying the requirement
````math
\operatorname{SpSt}(2n, 2k, ℝ)
    = \bigl\{ p ∈ ℝ^{2n × 2n} \, \big| \, p^TQ_{2n}p = Q_{2k} \bigr\},
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

Internally the dimensionality of the structure is stored as half of the even dimensions
supplied to the constructor, `SymplecticStiefel(2n, 2k) -> SymplecticStiefel{n, k, ℝ}()`,
as most computations with points on the Real symplectic Stiefel manifold takes
advantage of the natural block structure matrices
``A ∈ ℝ^{2n × 2k}`` where we consider it as consisting of four
smaller matrices in ``ℝ^{n × k}``.
"""
struct SymplecticStiefel{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}
end

@doc """
    SymplecticStiefel(two_n::Int, two_k::Int, field::AbstractNumbers=ℝ)
    -> SymplecticStiefel{div(two_n, 2), div(two_k, 2), ℝ}()

# Constructor:
The constructor for the [`Symplectic`](@ref) manifold accepts the even embedding
dimension ``n = 2k`` for the real symplectic manifold, ``ℝ^{2k × 2k}``.
"""
function SymplecticStiefel(two_n::Int, two_k::Int, field::AbstractNumbers=ℝ)
    return SymplecticStiefel{div(two_n, 2),div(two_k, 2),field}()
end

function Base.show(io::IO, ::SymplecticStiefel{n,k}) where {n,k}
    return print(io, "SymplecticStiefel{$(2n), $(2k)}()")
end

decorated_manifold(::SymplecticStiefel{n,k,ℝ}) where {n,k} = Euclidean(2n, 2k; field=ℝ)
ManifoldsBase.default_retraction_method(::SymplecticStiefel) = CayleyRetraction()
ManifoldsBase.default_inverse_retraction_method(::SymplecticStiefel) = CayleyInverseRetraction()

@doc raw"""
    manifold_dimension(::Symplectic{n})

Returns the dimension of the symplectic Stiefel manifold embedded in ``ℝ^{2n \times 2k}``,
i.e. [^Bendokat2021]
````math
    \operatorname{dim}(\operatorname{SpSt}(2n, 2k)) = (4n - 2k + 1)k.
````
"""
manifold_dimension(::SymplecticStiefel{n,k}) where {n,k} = (4n - 2k + 1) * k

@doc raw"""
    check_point(M::SymplecticStiefel, p; kwargs...)

Check whether `p` is a valid point on the [`SymplecticStiefel`](@ref),
$\operatorname{SpSt}(2n, 2k)$ manifold.
That is, the point has the right [`AbstractNumbers`](@ref) type and $p^{+}p$ is
(approximately) the identity,
where for $A \in \mathbb{R}^{2n \times 2k}$,
$A^{+} = Q_{2k}^TA^TQ_{2n}$ is the symplectic inverse, with
````math
Q_{2n} =
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n
\end{bmatrix}.
````
The tolerance can be set with `kwargs...` (e.g. `atol = 1.0e-14`).
"""
function check_point(M::SymplecticStiefel{n,k}, p; kwargs...) where {n,k}
    abstract_embedding_type = supertype(typeof(M))

    mpv = invoke(check_point, Tuple{abstract_embedding_type,typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv

    # Perform check that the matrix lives on the real symplectic manifold:
    expected_zero = norm(inv(M, p) * p - I)
    if !isapprox(expected_zero, 0; kwargs...)
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

@doc raw"""
    check_vector(M::Symplectic, p, X; kwargs...)

Checks whether `X` is a valid tangent vector at `p` on the [`SymplecticStiefel`](@ref),
``\operatorname{SpSt}(2n, 2k)`` manifold. First recall the definition of the symplectic
inverse for $A \in \mathbb{R}^{2n \times 2k}$,
$A^{+} = Q_{2k}^TA^TQ_{2n}$ is the symplectic inverse, with
````math
    Q_{2n} =
    \begin{bmatrix}
    0_n & I_n \\
     -I_n & 0_n
\end{bmatrix}.
````
The we check that
``(p^{+}X) = H \in 𝔤_{2k}`` approximately, where ``𝔤``
is the Lie Algebra of the symplectic group ``\operatorname{Sp}(2k)``,
characterized as [^Bendokat2021],
````math
    𝔤_{2k} = \{H \in ℝ^{2k \times 2k} \;|\; H^+ = -H \}.
````
The tolerance can be set with `kwargs...` (e.g. `atol = 1.0e-14`).
"""
check_vector(::SymplecticStiefel, ::Any...)

function check_vector(M::SymplecticStiefel{n,k,field}, p, X; kwargs...) where {n,k,field}
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

    # From Bendokat-Zimmermann: T_pSpSt(2n, 2k) = \{p*H | H^{+} = -H  \}

    H = inv(M, p) * X  # ∈ ℝ^{2k × 2k}, should be Hamiltonian.
    H_star = inv(Symplectic(2k, field), H)
    hamiltonian_identity_norm = norm(H + H_star)

    if !isapprox(hamiltonian_identity_norm, 0; kwargs...)
        return DomainError(
            hamiltonian_identity_norm,
            (
                "The matrix X is not in the tangent space at point p of the" *
                " $(M) manifold, as p^{+}X is not a Hamiltonian matrix."
            ),
        )
    end
    return nothing
end

@doc raw"""
    inner(M::SymplecticStiefel{n, k}, p, X. Y)

Compute the Riemannian inner product ``g^{\operatorname{SpSt}}`` at
``p \in \operatorname{SpSt}`` between tangent vectors ``X, X \in T_p\operatorname{SpSt}``.
Given by Proposition 3.10 of Benodkat-Zimmermann [^Bendokat2021].
````math
g^{\operatorname{SpSt}}_p(X, Y)
    = \operatorname{tr}\left(X^T\left(I_{2n} -
        \frac{1}{2}Q_{2n}^Tp(p^Tp)^{-1}p^TQ_{2n}\right)Y(p^Tp)^{-1}\right).
````
"""
function inner(::SymplecticStiefel{n,k}, p, X, Y) where {n,k}
    Q = SymplecticMatrix(p, X, Y)
    # Procompute lu(p'p) since we solve a^{-1}* 3 times
    a = lu(p' * p) # note that p'p is symmetric, thus so is its inverse c=a^{-1}
    b = Q' * p
    # we split the original trace into two one with I->(X'Yc)
    # and the other with 1/2 X'b c b' Y c
    # a) we permute X' and Y c to c^TY^TX = a\(Y'X) (avoids a large interims matrix)
    # b) we permute Y c up front, the center term is symmetric, so we get cY'b c b' X
    # and (b'X) again avoids a large interims matrix, so does Y'b.
    return tr(a \ (Y' * X)) - (1 / 2) * tr(a \ ((Y' * b) * (a \ (b' * X))))
end

@doc raw"""
    inv(::SymplecticStiefel{n, k}, A)
    inv!(::SymplecticStiefel{n, k}, q, p)

Compute the symplectic inverse ``A^+`` of matrix ``A ∈ ℝ^{2n × 2k}``. Given a matrix
````math
A ∈ ℝ^{2n × 2k},\quad
A =
\begin{bmatrix}
A_{1, 1} & A_{1, 2} \\
A_{2, 1} & A_{2, 2}
\end{bmatrix},\; A_{i, j} \in ℝ^{2n × 2k}
````
the symplectic inverse is defined as:
````math
A^{+} := Q_{2k}^T A^T Q_{2n},
````
where
````math
Q_{2n} =
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n
\end{bmatrix}.
````
For any ``p \in \operatorname{SpSt}(2n, 2k)``, ``p^{+}p = I_{2k}``.

The symplectic inverse of a matrix A can be expressed explicitly as:
````math
A^{+} =
\begin{bmatrix}
  A_{2, 2}^T & -A_{1, 2}^T \\[1.2mm]
 -A_{2, 1}^T &  A_{1, 1}^T
\end{bmatrix}.
````
"""
function Base.inv(M::SymplecticStiefel{n,k}, p) where {n,k}
    q = similar(p')
    return inv!(M, q, p)
end

function inv!(::SymplecticStiefel{n,k}, q, p) where {n,k}
    checkbounds(q, 1:(2k), 1:(2n))
    checkbounds(p, 1:(2n), 1:(2k))
    @inbounds for i in 1:k, j in 1:n
        q[i, j] = p[j + n, i + k]
    end
    @inbounds for i in 1:k, j in 1:n
        q[i, j + n] = -p[j, i + k]
    end
    @inbounds for i in 1:k, j in 1:n
        q[i + k, j] = -p[j + n, i]
    end
    @inbounds for i in 1:k, j in 1:n
        q[i + k, j + n] = p[j, i]
    end
    return q
end

@doc raw"""
    make_metric_compatible!(::SymplecticStiefel, Y, ::EuclideanMetric, p, X)

TODO: Complete.
Given the Riemannian metric at ``p``, ``g_p \colon `` we compute a mapping
``c : T_p\operatorname{SpSt}(2n, 2k) \rightarrow ℝ^{2n \times 2k}``
Calculated myself, but tailored to the right-invariant inner product of Bendokat-Zimmermann.
Does not have the correct range to be a 'change_representer!'-mapping.
"""
function make_metric_compatible!(
    ::SymplecticStiefel{n,k},
    Y,
    ::EuclideanMetric,
    p,
    X,
) where {n,k}
    Q = SymplecticMatrix(p, X)

    pT_p = p' * p
    pT_Q = p' * Q
    inner_term = pT_Q' * (lu(pT_p) \ pT_Q)  # ∈ ℝ^{2n × 2n}

    divisor = lu!(Array(add_scaled_I!(lmul!(-0.5, inner_term), 1.0)'))
    mul!(Y, X, pT_p)
    ldiv!(divisor, Y)
    return Y
end

@doc raw"""
    rand(M::SymplecticStiefel{n, k})

Generate a random point ``p \in \operatorname{SpSt}(2n, 2k)``
by first generating a random symplectic matrix
``p_{\operatorname{Sp}} \in \operatorname{Sp}(2n)``,
and then projecting onto the Symplectic Stiefel manifold using the
[`canonical_projection`](@ref).
That is, ``p = π(p_{\operatorname{Sp}})``
"""
function Base.rand(M::SymplecticStiefel{n,k}, hamiltonian_norm=1/2) where {n,k}
    p_symplectic = rand(Symplectic(2n), hamiltonian_norm)
    canonical_projection(M, p_symplectic)
end

@doc raw"""
    rand(::SymplecticStiefel{n, k}, p)

The symplectic Stiefel tangent space at ``p`` can be parametrized as [^Bendokat2021]
````math
    T_p\operatorname{SpSt}(2n, 2k) = \{X = pΩ + p^sB \;|\;
        Ω ∈ ℝ^{2k × 2k}, Ω^+ = -Ω,
        p^s ∈ \operatorname{SpSt}(2n, 2(n- k)), B ∈ ℝ^{2(n-k) × 2k}, \},
````
where ``Ω \in `` is Hamiltonian and ``p^s`` means the symplectic complement of ``p`` s.t.
``p^{+}p^{s} = 0``.

To then generate random tangent vectors at ``p``, we set ``B = 0`` and generate a random
Hamiltonian matrix ``Ω``.
"""
function Base.rand(::SymplecticStiefel{n,k}, p::AbstractMatrix) where {n,k}
    Ω = rand_hamiltonian(Symplectic(2k))
    p * Ω
end

@doc raw"""
    canonical_projection(::SymplecticStiefel, p_Sp)
    canonical_projection!(::SymplecticStiefel{n,k}, p, p_Sp)

Define the canonical projection from ``\operatorname{Sp}(2n, 2n)`` onto
``\operatorname{SpSt}(2n, 2k)``, by projecting onto the first ``k`` columns
and the ``n + 1``'th onto the ``n + k``'th columns [^Bendokat2021].

It is assumed that the point ``p`` is on ``\operatorname{Sp}(2n, 2n)``.
"""
function canonical_projection(M::SymplecticStiefel{n,k}, p_Sp) where {n,k}
    p_SpSt = similar(p_Sp, (2n, 2k))
    return canonical_projection!(M, p_SpSt, p_Sp)
end

function canonical_projection!(::SymplecticStiefel{n,k}, p, p_Sp) where {n,k}
    p[:, (1:k)] .= p_Sp[:, (1:k)]
    p[:, ((k + 1):(2k))] .= p_Sp[:, ((n + 1):(n + k))]
    return p
end

# compute p^+q (which is 2kx2k) in place of A
function symplectic_inverse_times(M::SymplecticStiefel{n,k}, p, q) where {n,k}
    A = similar(p, (2k, 2k))
    return symplectic_inverse_times!(M, A, p, q)
end

function symplectic_inverse_times!(::SymplecticStiefel{n,k}, A, p, q) where {n,k}
    # we write p = [p1 p2; p3 p4] (and q, too), then
    p1 = @view(p[1:n, 1:k])
    p2 = @view(p[1:n, (k + 1):(2k)])
    p3 = @view(p[(n + 1):(2n), 1:k])
    p4 = @view(p[(n + 1):(2n), (k + 1):(2k)])
    q1 = @view(q[1:n, 1:k])
    q2 = @view(q[1:n, (k + 1):(2k)])
    q3 = @view(q[(n + 1):(2n), 1:k])
    q4 = @view(q[(n + 1):(2n), (k + 1):(2k)])
    A1 = @view(A[1:k, 1:k])
    A2 = @view(A[1:k, (k + 1):(2k)])
    A3 = @view(A[(k + 1):(2k), 1:k])
    A4 = @view(A[(k + 1):(2k), (k + 1):(2k)])
    mul!(A1, p4', q1)           # A1  = p4'q1
    mul!(A1, p2', q3, -1, 1)    # A1 -= p2'p3
    mul!(A2, p4', q2)           # A2  = p4'q2
    mul!(A2, p2', q4, -1, 1)    # A2 -= p2'q4
    mul!(A3, p1', q3)           # A3  = p1'q3
    mul!(A3, p3', q1, -1, 1)    # A3 -= p3'q1
    mul!(A4, p1', q4)           # A4  = p1'q4
    mul!(A4, p3', q2, -1, 1)    #A4  -= p3'q2
    return A
end

@doc raw"""
    Compute the exponential mapping from eq. 3.19 of Bendokat-Zimmermann.
"""
function exp!(M::SymplecticStiefel{n, k}, q, p, X) where {n, k}
    # Cannot alias 'q' and 'p'!

    Q = SymplecticMatrix(p, X)
    pT_p = lu!(p' * p) # ∈ ℝ^{2k × 2k}

    # Construct A-bar:
    C = (pT_p) \ X' # ∈ ℝ^{2k × 2n}

    # A_bar = Q * (p^T * C^T) * Q
    A_bar = rmul!(lmul!(Q, (p' * C')), Q)
    A_bar .+= C * p

    # Last term, use C-memory:
    rmul!(C, Q') # C*Q^T -> C
    C_QT = C

    # Subtract C*Q^T*p*(pT_p)^{-1}*Q from A_bar:
    A_bar .-= rmul!(rdiv!(C_QT*p, pT_p), Q)
    # A_bar is "star-skew symmetric" (A^+ = -A).

    # Have not used the q (2n × 2k) memory.
    # Construct H_bar using q-memory:
    mul!(q, Q, X) # q = Q*X
    rdiv!(q, pT_p)
    rmul!(q, Q)

    q .-= p * symplectic_inverse_times(M, p, q)
    # H_bar = q

    q .+= p*A_bar

    # Rename q -> Δ_bar.
    Δ_bar = q

    γ_1 = Δ_bar - p*symplectic_inverse_times(M, p, Δ_bar)
    γ = [γ_1 -p] # ∈ ℝ^{2n × 4k}

    λ_1 = lmul!(Q', p*Q)
    λ_2 = (rmul!(Q'*Δ_bar', Q) .-
            (1/2) .* symplectic_inverse_times(M, Δ_bar, p)*rmul!(Q'*p', Q))'
    λ = [λ_1 λ_2] # ∈ ℝ^{2n × 4k}

    Γ = [λ -γ] # ∈ ℝ^{2n × 8k}
    Λ = [γ  λ] # ∈ ℝ^{2n × 8k}

    # Then compute the matrix exponentials:
    q .= Γ*(exp(Λ' * Γ)[:, (4k+1):end])*(exp(λ'*γ)[:, (2k+1):end])
    return q
end

@doc raw"""
    retract!(::SymplecticStiefel, q, p, X, ::CayleyRetraction)

Define the Cayley retraction on the Symplectic Stiefel manifold.
Reduced requirements down to inverting an (2k × 2k) matrix.
Formula due to Bendokat-Zimmermann Proposition 5.2.

# TODO: Add formula from Bendokat-Zimmermann.

# We set (t=1), regulate by the norm of the tangent vector how far to move.
"""
function retract!(M::SymplecticStiefel{n,k}, q, p, X, ::CayleyRetraction) where {n,k}
    # Define intermediate matrices for later use:
    A = symplectic_inverse_times(M, p, X)

    H = X .- p * A  # Allocates (2n × 2k).

    # A = I - A/2 + H^{+}H/4:
    A .= (symplectic_inverse_times(M, H, H) ./ 4) .- (A ./ 2)
    Manifolds.add_scaled_I!(A, 1.0)

    # Reuse 'H' memory:
    H .= H .+ 2 .* p
    r = lu!(A)
    q .= (-).(p) .+ rdiv!(H, r)
    return q
end

@doc raw"""
    inverse_retract!(::SymplecticStiefel, q, p, X, ::CayleyInverseRetraction)

Compute the Cayley Inverse Retraction ``X = \mathcal{L}_p^{\operatorname{SpSt}}(q)``
such that the Cayley Retraction from ``p`` along ``X`` lands at ``q``, i.e.
``\mathcal{R}_p(X) = q`` [^Bendokat2021].

First, recall the definition the standard symplectic matrix
````math
Q =
\begin{bmatrix}
 0    &  I \\
-I    &  0
\end{bmatrix}
````
as well as the symplectic inverse of a matrix ``A``, ``A^{+} = Q^T A^T Q``.

For ``p, q ∈ \operatorname{SpSt}(2n, 2k, ℝ)`` then, we can then define the
inverse cayley retraction as long as the following matrices exist.
````math
    U = (I + p^+ q)^{-1} \in ℝ^{2k \times 2k},
    \quad
    V = (I + q^+ p)^{-1} \in ℝ^{2k \times 2k}.
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
function inverse_retract!(M::SymplecticStiefel, X, p, q, ::CayleyInverseRetraction)
    U_inv = lu!(add_scaled_I!(symplectic_inverse_times(M, p, q), 1))
    V_inv = lu!(add_scaled_I!(symplectic_inverse_times(M, q, p), 1))

    X .= 2 .* ((p / V_inv .- p / U_inv) + ((p .+ q) / U_inv) .- p)
    return X
end

function gradient(M::SymplecticStiefel, f, p, backend::RiemannianProjectionBackend)
    amb_grad = _gradient(f, p, backend.diff_backend)
    return grad_euclidean_to_manifold(M, p, amb_grad)
end

function gradient!(M::SymplecticStiefel, f, X, p, backend::RiemannianProjectionBackend)
    _gradient!(f, X, p, backend.diff_backend)
    return grad_euclidean_to_manifold!(M, X, p, X)
end

function grad_euclidean_to_manifold(M::SymplecticStiefel, p, ∇f_euc)
    return grad_euclidean_to_manifold!(M, similar(∇f_euc), p, ∇f_euc)
end

function grad_euclidean_to_manifold!(::SymplecticStiefel, ∇f_man, p, ∇f_euc)
    Q = SymplecticMatrix(p, ∇f_euc)
    Qp = Q * p                      # Allocated memory.
    mul!(∇f_man, ∇f_euc, (p' * p))  # Use the memory in ∇f_man.
    ∇f_man .+= Qp * (∇f_euc' * Qp)
    return ∇f_man
end

@doc raw"""
    project!(::Union{SymplecticStiefel, Symplectic}, Y, p, A)

Given a point ``p \in \operatorname{SpSt}(2n, 2k)``,
project an element ``A \in \mathbb{R}^{2n \times 2k}`` onto
the tangent space ``T_p\operatorname{SpSt}(2n, 2k)`` relative to
the euclidean metric of the embedding ``\mathbb{R}^{2n \times 2k}``.

That is, we find the element ``X \in T_p\operatorname{SpSt}(2n, 2k)``
which solves the constrained optimization problem

````math
    \operatorname{min}_{X \in \mathbb{R}^{2n \times 2k}} \frac{1}{2}||X - A||^2, \quad
    \text{s.t.}\;
    h(X) = X^T Q p + p^T Q X = 0,
````
where ``h : \mathbb{R}^{2n \times 2k} \rightarrow \operatorname{skew}(2k)`` defines
the restriction of ``X`` onto the tangent space ``T_p\operatorname{SpSt}(2n, 2k)``.
"""
function project!(::Union{SymplecticStiefel,Symplectic}, Y, p, A)
    Q = SymplecticMatrix(Y, p, A)
    Q_p = Q * p

    function h(X)
        XT_Q_p = X' * Q_p
        return XT_Q_p .- XT_Q_p'
    end

    # Solve for Λ (Lagrange mutliplier):
    pT_p = p' * p  # (2k × 2k)
    Λ = sylvester(pT_p, pT_p, h(A) ./ 2)

    Y[:, :] = A .- Q_p * (Λ .- Λ')
    return Y
end
