@doc raw"""
The Symplectic Stiefel manifold. Each element represent a Symplectic Subspace of ``ℝ^{2n × 2k}``.
"""
struct SymplecticStiefel{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

@doc raw"""
    You are given a manifold of embedding dimension two_n × two_p.
    # Tried to type the fields being stored in SymplecticStiefel as well.
"""
function SymplecticStiefel(two_n::Int, two_k::Int, field::AbstractNumbers=ℝ)
    return SymplecticStiefel{div(two_n, 2),div(two_k, 2),field}()
end

function Base.show(io::IO, ::SymplecticStiefel{n,k}) where {n,k}
    return print(io, "SymplecticStiefel{$(2n), $(2k)}()")
end

decorated_manifold(::SymplecticStiefel{n,k,ℝ}) where {n,k} = Euclidean(2n, 2k; field=ℝ)
ManifoldsBase.default_retraction_method(::SymplecticStiefel) = CayleyRetraction()

@doc raw"""
    manifold_dimension(::SymplecticStiefel{n, k})

As shown in proposition 3.1 in Bendokat-Zimmermann.
"""
manifold_dimension(::SymplecticStiefel{n,k}) where {n,k} = (4n - 2k + 1) * k

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
                " inverse composed with itself is not the identity. $(p)"
            ),
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
                "The matrix X: $X is not in the tangent space at point p: $p of the" *
                " $(M) manifold, as p^{+}X is not a Hamiltonian matrix."
            ),
        )
    end
    return nothing
end

@doc raw"""
    inner(M::SymplecticStiefel{n, k}, p, X. Y)

Based on the inner product in Proposition 3.10 of Benodkat-Zimmermann.
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
    return return tr(a \ (Y' * X)) - 1 / 2 * tr(a \ ((Y' * b) * (a \ (b' * X))))
end

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
    change_representer!(::SymplecticStiefel{n, k}, Y, ::EuclideanMetric, p, X)

Calculated myself, but tailored to the right-invariant inner product of Bendokat-Zimmermann
"""
function change_representer!(
    ::SymplecticStiefel{n,k},
    Y,
    ::EuclideanMetric,
    p,
    X,
) where {n,k}
    # Quite an ugly expression: Have checked and it seems to be working.
    Q = SymplecticMatrix(p, X)
    I = UniformScaling(2n)

    # Remove memory allocation:
    # A = factorize((I - (1/2) * Q' * p * ((p' * p) \ p') * Q)')

    Y .= (lu((I - (1 / 2) * Q' * p * ((p' * p) \ p') * Q)') \ X) * p' * p
    return Y
end

@doc raw"""
    rand(M::SymplecticStiefel{n, k})

Use the canonical projection by first generating a random symplectic matrix of the correct size,
and then projecting onto the Symplectic Stiefel manifold.
"""
function Base.rand(M::SymplecticStiefel{n,k}) where {n,k}
    begin
        canonical_projection(M, rand(Symplectic(2n)))
    end
end

@doc raw"""
    rand(::SymplecticStiefel{n, k}, p)

As based on the parametrization of the tangent space ``T_p\operatorname{SpSt}(n, k)`` found in Proposition 3.2
of Benodkat-Zimmermann. There they express the tangent space as ``X = pΩ + p^sB``, where ``Ω^+ = -Ω`` is Hamiltonian.
The notation ``p^s`` means the symplectic complement of ``p`` s.t. ``p^{+}p^{s} = 0``, and ``B ∈ ℝ^{2(n-k) × 2k}.
"""
function Base.rand(::SymplecticStiefel{n,k}, p) where {n,k}
    begin
        Ω = rand_hamiltonian(Symplectic(2k))
        p * Ω
    end
end

@doc raw"""
    π(::SymplecticStiefel{n, k}, p) where {n, k}

Define the canonical projection from ``\operatorname{Sp}(2n, 2n)`` onto ``\operatorname{SpSt}(2n, 2k)``,
by projecting onto the first ``k`` columns, and the ``n + 1``'th column onto the ``n + k``'th columns.

It is assumed that the point ``p`` is on ``\operatorname{Sp}(2n, 2n)``.

# As done in Bendokat Zimmermann in equation 3.2.
"""
function canonical_projection(M::SymplecticStiefel{n,k}, p) where {n,k}
    p_SpSt = similar(p, (2n, 2k))
    return canonical_projection!(M, p_SpSt, p)
end

function canonical_projection!(::SymplecticStiefel{n,k}, p_SpSt, p) where {n,k}
    p_SpSt[:, (1:k)] .= p[:, (1:k)]
    p_SpSt[:, ((k + 1):(2k))] .= p[:, ((n + 1):(n + k))]
    return p_SpSt
end

# compute p^+q (which is 2kx2k) in place of A
function symplectic_inverse_times(M::SymplecticStiefel{n,k}, p, q) where {n,k}
    A = similar(p, (2k, 2k))
    return symplectic_inverse_times!(M, A, p, q)
end
function symplectic_inverse_times!(::SymplecticStiefel{n,k}, A, p, q) where {n,k}
    checkbounds(q, 1:(2n), 1:(2k))
    checkbounds(p, 1:(2n), 1:(2k))
    checkbounds(A, 1:(2k), 1:(2k))
    # we write p = [p1 p2; p3 p4] (and similarly q and A), where
    # pi, qi are nxk and Ai is kxk Then the p^+q can be computed as
    A .= 0
    @inbounds for i in 1:k, j in 1:k, l in 1:n # Compute A1 = p4'q1 - p2'q3
        A[i, j] += p[n + l, k + i] * q[l, j] - p[l, k + i] * q[n + l, j]
    end
    @inbounds for i in 1:k, j in 1:k, l in 1:n # A2 = p4'q2 - p2'q4
        A[i, k + j] += p[n + l, k + i] * q[l, k + j] - p[l, k + i] * q[n + l, k + j]
    end
    @inbounds for i in 1:k, j in 1:k, l in 1:n # A3 = p1'q3 - p3'q1
        A[k + i, j] += p[l, i] * q[n + l, j] - p[n + l, i] * q[l, j]
    end
    @inbounds for i in 1:k, j in 1:k, l in 1:n # A4 = p1'q4 - p3'q2
        A[k + i, k + j] += p[l, i] * q[n + l, k + j] - p[n + l, i] * q[l, k + j]
    end
    return A
end

function retract_broken!(M::SymplecticStiefel{n,k}, q, p, X, ::CayleyRetraction) where {n,k}
    # DANGER: q is aliased with p when called like:
    #     retract!(p.M, o.x, o.x, -s * o.gradient, o.retraction_method)
    # Leads to error in Manopt.gradient_descent!().

    # Define intermediate matrices for later use:
    #A = inv(M, p) * X # 2k x 2k - writing this out explicitly, since this allocates a 2kx2n matrix.
    p_plus = inv(M, p)
    A = p_plus * X

    # Cannot overwrite 'q' when it can be aliased with p:
    q .= X .- p * A # H in BZ21

    # Johannes: I think we have a bug here.
    # Want to calculate: (H^+ * H)/4 - A/2 -> A.
    # Should be: mul!(A, inv(M, q), q, 0.25, -0.5)
    #A .= -A./2 .+ symplectic_inverse_times(M, q, q)./4 , i.e. -A/2 + H^+H/4
    mul!(A, p_plus, q, 0.25, -0.5) #-A/2 + H^+H/4

    q .= q .+ 2 .* p
    Manifolds.add_scaled_I!(A, 1.0)
    r = lu!(A)
    q .= (-).(p) .+ rdiv!(q, r)
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
    # A = inv(M, p) * X # 2k x 2k - writing this out explicitly, since this allocates a 2kx2n matrix.
    A = symplectic_inverse_times(M, p, X)

    H = X .- p * A  # Allocates (2n × 2k).

    # A .= I - A/2 + H^+H/4:
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
    return grad_euclidean_to_manifold(M, p, amb_grad)
end

function gradient!(M::SymplecticStiefel, f, X, p, backend::RiemannianProjectionBackend)
    _gradient!(f, X, p, backend.diff_backend)
    return grad_euclidean_to_manifold!(M, X, p, X)
end

function grad_euclidean_to_manifold(::SymplecticStiefel, p, ∇f_euc)
    Q = SymplecticMatrix(p, ∇f_euc)
    return ∇f_euc * (p' * p) .+ Q * p * (∇f_euc' * Q * p)
end

function grad_euclidean_to_manifold!(::SymplecticStiefel, ∇f_man, p, ∇f_euc)
    # Older type: Rewritten to avoid allocating (2n × 2n) matrices:
    Q = SymplecticMatrix(p, ∇f_euc)
    Qp = Q * p
    ∇f_man .= ∇f_euc * (p' * p) .+ Qp * (∇f_euc' * Qp)
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
    \operatorname{min}_{X \in \mathbb{R}^{2n \times 2k}} \frac{1}{2}||X - A||, \quad
    s.t. h(X) = X^T Q p + p^T Q X = 0,
````
where ``h : \mathbb{R}^{2n \times 2k} \rightarrow \operatorname{skew}(2k)`` defines
the restriction of ``X`` onto the tangent space ``T_p\operatorname{SpSt}(2n, 2k)``.
"""
function project!(::Union{SymplecticStiefel, Symplectic}, Y, p, A)
    Q = SymplecticMatrix(Y, p, A)
    h(X) = X' * (Q * p) + p' * (Q * X)

    # Solve for Λ (Lagrange mutliplier):
    pT_p = p'*p  # (2k × 2k)
    Λ = sylvester(pT_p, pT_p, h(A) ./ 2)

    Y[:, :] = A .- (Q * p) * (Λ .- Λ')
    return Y
end
