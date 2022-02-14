@doc raw"""
    SymplecticStiefel{n, k, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}

The symplectic Stiefel manifold consists of all
$2n × 2k, \; n \geq k$ matrices satisfying the requirement
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
\end{bmatrix}.
````

The symplectic Stiefel tangent space at ``p`` can be parametrized as [^Bendokat2021]
````math
    \begin{align*}
    T_p\operatorname{SpSt}(2n, 2k)
    = \{&X \in \mathbb{R}^{2n \times 2k} \;|\; p^{T}Q_{2n}X + X^{T}Q_{2n}p = 0 \}, \\
    = \{&X = pΩ + p^sB \;|\;
        Ω ∈ ℝ^{2k × 2k}, Ω^+ = -Ω, \\
        &\quad\quad\quad p^s ∈ \operatorname{SpSt}(2n, 2(n- k)), B ∈ ℝ^{2(n-k) × 2k}, \},
    \end{align*}
````
where ``Ω \in \mathfrak{sp}(2n,F)`` is Hamiltonian and ``p^s`` means
the symplectic complement of ``p`` s.t. ``p^{+}p^{s} = 0``.

# Constructor
    SymplecticStiefel(2n, 2k, field = ℝ)

Generate the (real-valued) symplectic Stiefel manifold of ``2n \times 2k``
matrices which span a ``2k`` dimensional symplectic subspace of ``ℝ^{2n \times 2n}``.
"""
struct SymplecticStiefel{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

@doc raw"""
    SymplecticStiefel(two_n::Int, two_k::Int, field::AbstractNumbers=ℝ)
    -> SymplecticStiefel{div(two_n, 2), div(two_k, 2), field}()

# Constructor:
The constructor for the [`SymplecticStiefel`](@ref) manifold accepts the even column
dimension ``n = 2m`` and an even number of columns ``k = 2l`` for
the real symplectic Stiefel manifold with elements ``p \in ℝ^{2k × 2k}``.
"""
function SymplecticStiefel(two_n::Int, two_k::Int, field::AbstractNumbers=ℝ)
    return SymplecticStiefel{div(two_n, 2),div(two_k, 2),field}()
end

decorated_manifold(::SymplecticStiefel{n,k,ℝ}) where {n,k} = Euclidean(2n, 2k; field=ℝ)

function ManifoldsBase.default_inverse_retraction_method(::SymplecticStiefel)
    return CayleyInverseRetraction()
end

ManifoldsBase.default_retraction_method(::SymplecticStiefel) = CayleyRetraction()

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
The we check that ``H = p^{+}X \in 𝔤_{2k}``, where ``𝔤``
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
    exp(::SymplecticStiefel, p, X)
    exp!(M::SymplecticStiefel, q, p, X)

Compute the exponential mapping
````math
    \operatorname{exp}\colon T\operatorname{SpSt}(2n, 2k)
    \rightarrow \operatorname{SpSt}(2n, 2k)
````
at a point ``p \in  \operatorname{SpSt}(2n, 2k)``
in the direction of ``X \in T_p\operatorname{SpSt}(2n, 2k)``.

The tangent vector ``X`` can be written in the form
``X = \bar{\Omega}p`` [^Bendokat2021], with
````math
    \bar{\Omega} = X (p^Tp)^{-1}p^T
        + Q_{2n}p(p^Tp)^{-1}X^T(I_{2n} - Q_{2n}^Tp(p^Tp)^{-1}p^TQ_{2n})Q_{2n}
        \in ℝ^{2n \times 2n},
````
where ``Q_{2n}`` is the [`SymplecticMatrix`](@ref). Using this expression for ``X``,
the exponential mapping can be computed as
````math
    \operatorname{exp}_p(X) = \operatorname{Exp}([\bar{\Omega} - \bar{\Omega}^T])
                              \operatorname{Exp}(\bar{\Omega}^T)p,
````
where ``\operatorname{Exp}(\cdot)`` denotes the matrix exponential.

Computing the above mapping directly however, requires taking matrix exponentials
of two ``2n \times 2n`` matrices, which is computationally expensive when ``n``
increases. Therefore we instead follow [^Bendokat2021] who express the above
exponential mapping in a way which only requires taking matrix exponentials
of an ``8k \times 8k`` matrix and a ``4k \times 4k`` matrix.

To this end, first define
````math
\bar{A} = Q_{2k}p^TX(p^Tp)^{-1}Q_{2k} +
            (p^Tp)^{-1}X^T(p - Q_{2n}^Tp(p^Tp)^{-1}Q_{2k}) \in ℝ^{2k \times 2k},
````
and
````math
\bar{H} = (I_{2n} - pp^+)Q_{2n}X(p^Tp)^{-1}Q_{2k} \in ℝ^{2n \times 2k}.
````
We then let ``\bar{\Delta} = p\bar{A} + \bar{H}``, and define the matrices
````math
    γ = \left[\left(I_{2n} - \frac{1}{2}pp^+\right)\bar{\Delta} \quad
              -p \right] \in ℝ^{2n \times 4k},
````
and
````math
    λ = \left[Q_{2n}^TpQ_{2k} \quad
        \left(\bar{\Delta}^+\left(I_{2n}
              - \frac{1}{2}pp^+\right)\right)^T\right] \in ℝ^{2n \times 4k}.
````
With the above defined matrices it holds that ``\bar{\Omega} = λγ^T``.
 As a last preliminary step, concatenate ``γ`` and ``λ`` to define the matrices
``Γ = [λ \quad -γ] \in ℝ^{2n \times 8k}`` and
``Λ = [γ \quad λ] \in ℝ^{2n \times 8k}``.

With these matrix constructions done, we can compute the
exponential mapping as
````math
    \operatorname{exp}_p(X) =
        Γ \operatorname{Exp}(ΛΓ^T)
        \begin{bmatrix}
            0_{4k} \\
            I_{4k}
        \end{bmatrix}
        \operatorname{Exp}(λγ^T)
        \begin{bmatrix}
            0_{2k} \\
            I_{2k}
        \end{bmatrix}.
````
which only requires computing the matrix exponentials of
``ΛΓ^T \in ℝ^{8k \times 8k}`` and ``λγ^T \in ℝ^{4k \times 4k}``.
"""
exp(::SymplecticStiefel, p, X)

function exp!(M::SymplecticStiefel{n,k}, q, p, X) where {n,k}
    Q = SymplecticMatrix(p, X)
    pT_p = lu(p' * p) # ∈ ℝ^{2k × 2k}

    C = pT_p \ X' # ∈ ℝ^{2k × 2n}

    # Construct A-bar:
    # First A-term: Q * (p^T * C^T) * Q
    A_bar = rmul!(lmul!(Q, (p' * C')), Q)
    A_bar .+= C * p

    # Second A-term, use C-memory:
    rmul!(C, Q') # C*Q^T -> C
    C_QT = C

    # Subtract C*Q^T*p*(pT_p)^{-1}*Q:
    # A_bar is "star-skew symmetric" (A^+ = -A).
    A_bar .-= rmul!((C_QT * p) / pT_p, Q)

    # Construct H_bar:
    # First H-term: Q * (C_QT * Q)' * Q -> Q * C' * Q = Q * (X / pT_p) * Q
    H_bar = rmul!(lmul!(Q, rmul!(C_QT, Q)'), Q)
    # Subtract second H-term:
    H_bar .-= p * symplectic_inverse_times(M, p, H_bar)

    # Construct Δ_bar in H_bar-memory:
    H_bar .+= p * A_bar

    # Rename H_bar -> Δ_bar.
    Δ_bar = H_bar

    γ_1 = Δ_bar - (1 / 2) .* p * symplectic_inverse_times(M, p, Δ_bar)
    γ = [γ_1 -p] # ∈ ℝ^{2n × 4k}

    Δ_bar_star = rmul!(Q' * Δ_bar', Q)
    λ_1 = lmul!(Q', p * Q)
    λ_2 = (Δ_bar_star .- (1 / 2) .* (Δ_bar_star * p) * λ_1')'
    λ = [λ_1 λ_2] # ∈ ℝ^{2n × 4k}

    Γ = [λ -γ] # ∈ ℝ^{2n × 8k}
    Λ = [γ λ] # ∈ ℝ^{2n × 8k}

    # At last compute the (8k × 8k) and (4k × 4k) matrix exponentials:
    q .= Γ * (exp(Λ' * Γ)[:, (4k + 1):end]) * (exp(λ' * γ)[:, (2k + 1):end])
    return q
end

@doc raw"""
    gradient(::SymplecticStiefel, f, p, backend::RiemannianProjectionBackend)
    gradient!(::SymplecticStiefel, f, X, p, backend::RiemannianProjectionBackend)

Compute the gradient of
``f\colon \operatorname{SpSt}(2n, 2k) \rightarrow ℝ``
at ``p \in \operatorname{SpSt}(2n, 2k)``.

We first compute the embedding gradient ``∇f(p) \in ℝ^{2n \times 2k}`` using
the [`AbstractRiemannianDiffBackend`](@ref) in the [`RiemannianProjectionBackend`](@ref).
Then we transform the embedding gradient to the unique tangent vector space element
``\text{grad}f(p) \in T_p\operatorname{SpSt}(2n, 2k)``
which fulfills the variational equation
````math
    g_p(\text{grad}f(p), X)
    = \text{D}f(p)
    = \langle ∇f(p), X \rangle
    \quad\forall\; X \in T_p\operatorname{SpSt}(2n, 2k).
````
The manifold gradient ``\text{grad}f(p)`` is computed from ``∇f(p)`` as
````math
    \text{grad}f(p) = ∇f(p)p^Tp + Q_{2n}p∇f(p)^TQ_{2n}p,
````
where ``Q_{2n}`` is the [`SymplecticMatrix`](@ref).
"""
function gradient(M::SymplecticStiefel, f, p, backend::RiemannianProjectionBackend)
    amb_grad = _gradient(f, p, backend.diff_backend)
    return grad_euclidean_to_manifold(M, p, amb_grad)
end

function gradient!(M::SymplecticStiefel, f, X, p, backend::RiemannianProjectionBackend)
    _gradient!(f, X, p, backend.diff_backend)
    return grad_euclidean_to_manifold!(M, X, p, copy(X))
end

function grad_euclidean_to_manifold(M::SymplecticStiefel, p, ∇f_euc)
    return grad_euclidean_to_manifold!(M, similar(∇f_euc), p, ∇f_euc)
end

function grad_euclidean_to_manifold!(::SymplecticStiefel, ∇f_man, p, ∇f_euc)
    # ∇f_man cannot be aliased with ∇f_euc:
    if ∇f_man === ∇f_euc
        throw(
            ArgumentError(
                "Output gradient '∇f_man' must not be aliased with input gradient '∇f_euc'.",
            ),
        )
    end
    Q = SymplecticMatrix(p, ∇f_euc)
    Qp = Q * p                      # Allocated memory.
    mul!(∇f_man, ∇f_euc, (p' * p))  # Use the memory in ∇f_man.
    ∇f_man .+= Qp * (∇f_euc' * Qp)
    return ∇f_man
end

@doc raw"""
    inner(M::SymplecticStiefel{n, k}, p, X. Y)

Compute the Riemannian inner product ``g^{\operatorname{SpSt}}`` at
``p \in \operatorname{SpSt}`` between tangent vectors ``X, X \in T_p\operatorname{SpSt}``.
Given by Proposition 3.10 in [^Bendokat2021].
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
For any ``p \in \operatorname{SpSt}(2n, 2k)`` we have that ``p^{+}p = I_{2k}``.

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
    inverse_retract(::SymplecticStiefel, p, q, ::CayleyInverseRetraction)
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

For ``p, q ∈ \operatorname{SpSt}(2n, 2k, ℝ)`` then, we can define the
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
inverse_retract(::SymplecticStiefel, p, q, ::CayleyInverseRetraction)

function inverse_retract!(M::SymplecticStiefel, X, p, q, ::CayleyInverseRetraction)
    U_inv = lu!(add_scaled_I!(symplectic_inverse_times(M, p, q), 1))
    V_inv = lu!(add_scaled_I!(symplectic_inverse_times(M, q, p), 1))

    X .= 2 .* ((p / V_inv .- p / U_inv) .+ ((p + q) / U_inv) .- p)
    return X
end

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
    project(::Union{SymplecticStiefel,Symplectic}, p, A)
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
project(::Union{SymplecticStiefel,Symplectic}, p, A)

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

@doc raw"""
    rand(M::SymplecticStiefel; vector_at=nothing,
        hamiltonian_norm=(vector_at === nothing ? 1/2 : 1.0))

Generate a random point ``p \in \operatorname{SpSt}(2n, 2k)`` or
a random tangent vector ``X \in T_p\operatorname{SpSt}(2n, 2k)``
if `vector_at` is set to a point ``p \in \operatorname{Sp}(2n)``.

A random point on ``\operatorname{SpSt}(2n, 2k)`` is found by first generating a
random point on the symplectic manifold ``\operatorname{Sp}(2n)``,
and then projecting onto the Symplectic Stiefel manifold using the
[`canonical_projection`](@ref) ``π_{\operatorname{SpSt}(2n, 2k)}``.
That is, ``p = π_{\operatorname{SpSt}(2n, 2k)}(p_{\operatorname{Sp}})``.

To generate a random tangent vector in ``T_p\operatorname{SpSt}(2n, 2k)``
this code exploits the second tangent vector space parametrization of
[`SymplecticStiefel`](@ref), showing that any ``X \in T_p\operatorname{SpSt}(2n, 2k)``
can be written as ``X = pΩ_X + p^sB_X``.
To generate random tangent vectors at ``p`` then, this function sets ``B_X = 0``
and generates a random Hamiltonian matrix ``Ω_X \in \mathfrak{sp}(2n,F)`` with
Frobenius norm of `hamiltonian_norm` before returning ``X = pΩ_X``.
"""
function Base.rand(
    M::SymplecticStiefel{n};
    vector_at=nothing,
    hamiltonian_norm=(vector_at === nothing ? 1 / 2 : 1.0),
) where {n}
    if vector_at === nothing
        return canonical_projection(
            M,
            rand(Symplectic(2n); hamiltonian_norm=hamiltonian_norm),
        )# code for generation of random points
    else
        return random_vector(M, vector_at; hamiltonian_norm=hamiltonian_norm)
    end
end

function random_vector(
    ::SymplecticStiefel{n,k},
    p::AbstractMatrix;
    hamiltonian_norm=1.0,
) where {n,k}
    Ω = rand_hamiltonian(Symplectic(2k); frobenius_norm=hamiltonian_norm)
    return p * Ω
end

@doc raw"""
    retract(::SymplecticStiefel, p, X, ::CayleyRetraction)
    retract!(::SymplecticStiefel, q, p, X, ::CayleyRetraction)

Compute the Cayley retraction on the Symplectic Stiefel manifold, computed inplace
of `q` from `p` along `X`.

Given a point ``p \in \operatorname{SpSt}(2n, 2k)``, every tangent vector
``X \in T_p\operatorname{SpSt}(2n, 2k)`` is of the form
``X = \tilde{\Omega}p``, with
````math
    \tilde{\Omega} = \left(I_{2n} - \frac{1}{2}pp^+\right)Xp^+ -
                     pX^+\left(I_{2n} - \frac{1}{2}pp^+\right) \in ℝ^{2n \times 2n},
````
as shown in Proposition 3.5 of [^Bendokat2021].
Using this representation of ``X``, the Cayley retraction
on ``\operatorname{SpSt}(2n, 2k)`` is defined pointwise as
````math
    \mathcal{R}_p(X) = \operatorname{cay}\left(\frac{1}{2}\tilde{\Omega}\right)p.
````
The operator ``\operatorname{cay}(A) = (I - A)^{-1}(I + A)`` is the Cayley transform.

However, the computation of an ``2n \times 2n`` matrix inverse in the expression
above can be reduced down to inverting a ``2k \times 2k`` matrix due to Proposition
5.2 of [^Bendokat2021].

Let ``A = p^+X`` and ``H = X - pA``. Then an equivalent expression for the Cayley
retraction defined pointwise above is
````math
    \mathcal{R}_p(X) = -p + (H + 2p)(H^+H/4 - A/2 + I_{2k})^{-1}.
````
It is this expression we compute inplace of `q`.
"""
retract(::SymplecticStiefel, p, X, ::CayleyRetraction)

function retract!(M::SymplecticStiefel, q, p, X, ::CayleyRetraction)
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

function Base.show(io::IO, ::SymplecticStiefel{n,k,𝔽}) where {n,k,𝔽}
    return print(io, "SymplecticStiefel{$(2n), $(2k), $(𝔽)}()")
end

@doc raw"""
    symplectic_inverse_times(::SymplecticStiefel, p, q)
    symplectic_inverse_times!(::SymplecticStiefel, A, p, q)

Directly compute the symplectic inverse of ``p \in \operatorname{SpSt}(2n, 2k)``,
multiplied with ``q \in \operatorname{SpSt}(2n, 2k)``.
That is, this function efficiently computes
``p^+q = (Q_{2k}p^TQ_{2n})q \in ℝ^{2k \times 2k}``,
where ``Q_{2n}, Q_{2k}`` are the [`SymplecticMatrix`](@ref)
of sizes ``2n \times 2n`` and ``2k \times 2k`` respectively.

This function performs this common operation without allocating more than
a ``2k \times 2k`` matrix to store the result in, or in the case of the in-place
function, without allocating memory at all.
"""
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
