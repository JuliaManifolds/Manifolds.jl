@doc raw"""
    SymplecticStiefel{T,𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}

The symplectic Stiefel manifold consists of all
``2n×2k, n ≥ k`` matrices satisfying the requirement

````math
\mathrm{SpSt}(2n, 2k, ℝ)
    := \bigl\{ p ∈ ℝ^{2n×2n} \ \big| \ p^{\mathrm{T}}J_{2n}p = J_{2k} \bigr\},
````

where ``J_{2n}`` denotes the [`SymplecticElement`](@ref)

````math
J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}.
````

The symplectic Stiefel tangent space at ``p`` can be parametrized as [BendokatZimmermann:2021](@cite)

```math
\begin{align*}
    T_p\mathrm{SpSt}(2n, 2k)
    &= \{X ∈ ℝ^{2n×2k} ∣ p^{T}J_{2n}X + X^{T}J_{2n}p = 0 \}, \\
    &= \{X = pΩ + p^sB \mid
        Ω ∈ ℝ^{2k×2k}, Ω^+ = -Ω, \\
        &\quad\qquad p^s ∈ \mathrm{SpSt}(2n, 2(n- k)), B ∈ ℝ^{2(n-k)×2k}, \},
\end{align*}
```

where ``Ω ∈ \mathfrak{sp}(2n,F)`` is [`Hamiltonian`](@ref) and ``p^s`` means
the symplectic complement of ``p`` s.t. ``p^{+}p^{s} = 0``.
Here ``p^+`` denotes the [`symplectic_inverse`](@ref).

You can also use [`StiefelPoint`](@ref) and [`StiefelTVector`](@ref) with this manifold,
they are equivalent to using arrays.

# Constructor
    SymplecticStiefel(2n::Int, 2k::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:type)

Generate the (real-valued) symplectic Stiefel manifold of ``2n×2k``
matrices which span a ``2k`` dimensional symplectic subspace of ``ℝ^{2n×2n}``.
The constructor for the [`SymplecticStiefel`](@ref) manifold accepts the even column
dimension ``2n`` and an even number of columns ``2k`` for
the real symplectic Stiefel manifold with elements ``p ∈ ℝ^{2n×2k}``.
"""
struct SymplecticStiefel{T,𝔽} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function SymplecticStiefel(
    two_n::Int,
    two_k::Int,
    field::AbstractNumbers=ℝ;
    parameter::Symbol=:type,
)
    two_n % 2 == 0 || throw(
        ArgumentError(
            "The first matrix size of the symplectic Stiefel manifold must be even, but was $(two_n).",
        ),
    )
    two_k % 2 == 0 || throw(
        ArgumentError(
            "The second matrix size of the symplectic Stiefel manifold must be even. but was $(two_k).",
        ),
    )
    size = wrap_type_parameter(parameter, (div(two_n, 2), div(two_k, 2)))
    return SymplecticStiefel{typeof(size),field}(size)
end

function active_traits(f, ::SymplecticStiefel, args...)
    return merge_traits(IsEmbeddedManifold(), IsDefaultMetric(RealSymplecticMetric()))
end

# Define Stiefel as the array fallback
ManifoldsBase.@default_manifold_fallbacks SymplecticStiefel StiefelPoint StiefelTVector value value

function ManifoldsBase.default_inverse_retraction_method(::SymplecticStiefel)
    return CayleyInverseRetraction()
end

ManifoldsBase.default_retraction_method(::SymplecticStiefel) = CayleyRetraction()

@doc raw"""
    canonical_project(::SymplecticStiefel, p_Sp)
    canonical_project!(::SymplecticStiefel, p, p_Sp)

Define the canonical projection from ``\mathrm{Sp}(2n, 2n)`` onto
``\mathrm{SpSt}(2n, 2k)``, by projecting onto the first ``k`` columns
and the ``n + 1``'th onto the ``n + k``'th columns [BendokatZimmermann:2021](@cite).

It is assumed that the point ``p`` is on ``\mathrm{Sp}(2n, 2n)``.
"""
function canonical_project(M::SymplecticStiefel, p_Sp)
    n, k = get_parameter(M.size)
    p_SpSt = similar(p_Sp, (2n, 2k))
    return canonical_project!(M, p_SpSt, p_Sp)
end

function canonical_project!(M::SymplecticStiefel, p, p_Sp)
    n, k = get_parameter(M.size)
    p[:, (1:k)] .= p_Sp[:, (1:k)]
    p[:, ((k + 1):(2k))] .= p_Sp[:, ((n + 1):(n + k))]
    return p
end

@doc raw"""
    check_point(M::SymplecticStiefel, p; kwargs...)

Check whether `p` is a valid point on the [`SymplecticStiefel`](@ref),
``\mathrm{SpSt}(2n, 2k)`` manifold, that is ``p^{+}p`` is the identity,
``(⋅)^+`` denotes the [`symplectic_inverse`](@ref).
"""
function check_point(M::SymplecticStiefel{<:Any,ℝ}, p; kwargs...)
    # Perform check that the matrix lives on the real symplectic manifold:
    if !isapprox(inv(M, p) * p, I; kwargs...)
        return DomainError(
            norm(inv(M, p) * p - I),
            (
                "The point p does not lie on $(M) because its symplectic" *
                " inverse composed with itself is not the identity."
            ),
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::SymplecticMatrices, p, X; kwargs...)

Checks whether `X` is a valid tangent vector at `p` on the [`SymplecticStiefel`](@ref),
``\mathrm{SpSt}(2n, 2k)`` manifold.

The check consists of verifying that ``H = p^{+}X ∈ 𝔤_{2k}``, where ``𝔤``
is the Lie Algebra of the symplectic group ``\mathrm{Sp}(2k)``, that is
the set of [`HamiltonianMatrices`])(@ref), where ``(⋅)^+`` denotes the [`symplectic_inverse`](@ref).
"""
check_vector(::SymplecticStiefel, ::Any...)

function check_vector(M::SymplecticStiefel{S,𝔽}, p, X::T; kwargs...) where {S,T,𝔽}
    n, k = get_parameter(M.size)
    # From Bendokat-Zimmermann: T_pSpSt(2n, 2k) = \{p*H | H^{+} = -H  \}
    H = inv(M, p) * X  # ∈ ℝ^{2k×2k}, should be Hamiltonian.

    if !is_hamiltonian(H; kwargs...)
        return DomainError(
            norm(Matrix(Hamiltonian(H)^+) + H),
            (
                "The matrix X is not in the tangent space at point p of $M at $p, since p^{+}X is not a Hamiltonian matrix."
            ),
        )
    end
    return nothing
end

@doc raw"""
    exp(::SymplecticStiefel, p, X)
    exp!(M::SymplecticStiefel, q, p, X)

Compute the exponential mapping
```math
  \exp\colon T\mathrm{SpSt}(2n, 2k) → \mathrm{SpSt}(2n, 2k)
```
at a point ``p ∈ \mathrm{SpSt}(2n, 2k)``
in the direction of ``X ∈ T_p\mathrm{SpSt}(2n, 2k)``.

The tangent vector ``X`` can be written in the form
``X = \bar{\Omega}p`` [BendokatZimmermann:2021](@cite), with

```math
  \bar{\Omega} = X (p^{\mathrm{T}}p)^{-1}p^{\mathrm{T}}
    + J_{2n}p(p^{\mathrm{T}}p)^{-1}X^{\mathrm{T}}(I_{2n}
    - J_{2n}^{\mathrm{T}}p(p^{\mathrm{T}}p)^{-1}p^{\mathrm{T}}J_{2n})J_{2n}
    ∈ ℝ^{2n×2n},
```

where ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).

Using this expression for ``X``,
the exponential mapping can be computed as

````math
  \exp_p(X) = \operatorname{Exp}([\bar{\Omega} - \bar{\Omega}^{\mathrm{T}}])
                             \operatorname{Exp}(\bar{\Omega}^{\mathrm{T}})p,
````
where ``\operatorname{Exp}(⋅)`` denotes the matrix exponential.

Computing the above mapping directly however, requires taking matrix exponentials
of two ``2n×2n`` matrices, which is computationally expensive when ``n``
increases. Therefore we instead follow [BendokatZimmermann:2021](@cite) who express the above
exponential mapping in a way which only requires taking matrix exponentials
of an ``8k×8k`` matrix and a ``4k×4k`` matrix.

To this end, first define
```math
\bar{A} = J_{2k}p^{\mathrm{T}}X(p^{\mathrm{T}}p)^{-1}J_{2k} +
            (p^{\mathrm{T}}p)^{-1}X^{\mathrm{T}}(p - J_{2n}^{\mathrm{T}}p(p^{\mathrm{T}}p)^{-1}J_{2k}) ∈ ℝ^{2k×2k},
```

and

```math
\bar{H} = (I_{2n} - pp^+)J_{2n}X(p^{\mathrm{T}}p)^{-1}J_{2k} ∈ ℝ^{2n×2k}.
```

We then let ``\bar{\Delta} = p\bar{A} + \bar{H}``, and define the matrices

```math
    γ = \left[\left(I_{2n} - \frac{1}{2}pp^+\right)\bar{\Delta} \quad
              -p \right] ∈ ℝ^{2n×4k},
```
and
````math
    λ = \left[J_{2n}^{\mathrm{T}}pJ_{2k} \quad
        \left(\bar{\Delta}^+\left(I_{2n}
              - \frac{1}{2}pp^+\right)\right)^{\mathrm{T}}\right] ∈ ℝ^{2n×4k}.
````
With the above defined matrices it holds that ``\bar{\Omega} = λγ^{\mathrm{T}}``.
 As a last preliminary step, concatenate ``γ`` and ``λ`` to define the matrices
``Γ = [λ \quad -γ] ∈ ℝ^{2n×8k}`` and
``Λ = [γ \quad λ] ∈ ℝ^{2n×8k}``.

With these matrix constructions done, we can compute the
exponential mapping as
```math
  \exp_p(X) = Γ \operatorname{Exp}(ΛΓ^{\mathrm{T}})
    \begin{bmatrix} 0_{4k} \\ I_{4k} \end{bmatrix}
    \operatorname{Exp}(λγ^{\mathrm{T}})
    \begin{bmatrix} 0_{2k} \\ I_{2k} \end{bmatrix}.
```

which only requires computing the matrix exponentials of
``ΛΓ^{\mathrm{T}} ∈ ℝ^{8k×8k}`` and ``λγ^{\mathrm{T}} ∈ ℝ^{4k×4k}``.
"""
exp(::SymplecticStiefel, p, X)

function exp!(M::SymplecticStiefel, q, p, X)
    n, k = get_parameter(M.size)
    J = SymplecticElement(p, X)
    pT_p = lu(p' * p) # ∈ ℝ^{2k×2k}

    C = pT_p \ X' # ∈ ℝ^{2k×2n}

    # Construct A-bar:
    # First A-term: J * (p^{\mathrm{T}} * C^{\mathrm{T}}) * J
    A_bar = rmul!(lmul!(J, (p' * C')), J)
    A_bar .+= C * p

    # Second A-term, use C-memory:
    rmul!(C, J') # C*J^{\mathrm{T}} -> C
    C_JT = C

    # Subtract C*J^{\mathrm{T}}*p*(pT_p)^{-1}*J:
    # A_bar is "star-skew symmetric" (A^+ = -A).
    A_bar .-= rmul!((C_JT * p) / pT_p, J)

    # Construct H_bar:
    # First H-term: J * (C_JT * J)' * J -> J * C' * J = J * (X / pT_p) * J
    H_bar = rmul!(lmul!(J, rmul!(C_JT, J)'), J)
    # Subtract second H-term:
    H_bar .-= p * symplectic_inverse_times(M, p, H_bar)

    # Construct Δ_bar in H_bar-memory:
    H_bar .+= p * A_bar

    # Rename H_bar -> Δ_bar.
    Δ_bar = H_bar

    γ_1 = Δ_bar - (1 / 2) .* p * symplectic_inverse_times(M, p, Δ_bar)
    γ = [γ_1 -p] # ∈ ℝ^{2n×4k}

    Δ_bar_star = rmul!(J' * Δ_bar', J)
    λ_1 = lmul!(J', p * J)
    λ_2 = (Δ_bar_star .- (1 / 2) .* (Δ_bar_star * p) * λ_1')'
    λ = [λ_1 λ_2] # ∈ ℝ^{2n×4k}

    Γ = [λ -γ] # ∈ ℝ^{2n×8k}
    Λ = [γ λ] # ∈ ℝ^{2n×8k}

    # At last compute the (8k×8k) and (4k×4k) matrix exponentials:
    q .= Γ * (exp(Λ' * Γ)[:, (4k + 1):end]) * (exp(λ' * γ)[:, (2k + 1):end])
    return q
end

function get_embedding(::SymplecticStiefel{TypeParameter{Tuple{n,k}},𝔽}) where {n,k,𝔽}
    return Euclidean(2 * n, 2 * k; field=𝔽)
end
function get_embedding(M::SymplecticStiefel{Tuple{Int,Int},𝔽}) where {𝔽}
    n, k = get_parameter(M.size)
    return Euclidean(2 * n, 2 * k; field=𝔽, parameter=:field)
end

@doc raw"""
    get_total_space(::SymplecticStiefel)

Return the total space of the [`SymplecticStiefel`](@ref) manifold, which is the corresponding [`SymplecticMatrices`](@ref) manifold.
"""
function get_total_space(::SymplecticStiefel{TypeParameter{Tuple{n,k}},ℝ}) where {n,k}
    return SymplecticMatrices(2 * n)
end
function get_total_space(M::SymplecticStiefel{Tuple{Int,Int},ℝ})
    n, _ = get_parameter(M.size)
    return SymplecticMatrices(2 * n; parameter=:field)
end

@doc raw"""
    inner(M::SymplecticStiefel, p, X. Y)

Compute the Riemannian inner product ``g^{\mathrm{SpSt}}`` at
``p ∈ \mathrm{SpSt}`` of tangent vectors ``Y, X ∈ T_p\mathrm{SpSt}``.
Given by Proposition 3.10 in [BendokatZimmermann:2021](@cite).
```math
g^{\mathrm{SpSt}}_p(X, Y)
  = \operatorname{tr}\Bigl(
    X^{\mathrm{T}}\bigl(
      I_{2n} - \frac{1}{2}J_{2n}^{\mathrm{T}} p(p^{\mathrm{T}}p)^{-1}p^{\mathrm{T}}J_{2n}
    \bigr) Y (p^{\mathrm{T}}p)^{-1}\Bigr).
```
"""
function inner(::SymplecticStiefel, p, X, Y)
    J = SymplecticElement(p, X, Y) # in BZ21 also J
    # Procompute lu(p'p) since we solve a^{-1}* 3 times
    a = lu(p' * p) # note that p'p is symmetric, thus so is its inverse c=a^{-1}
    b = J' * p
    # we split the original trace into two one with I->(X'Yc)
    # and the other with 1/2 X'b c b' Y c
    # 1) we permute X' and Y c to c^{\mathrm{T}}Y^{\mathrm{T}}X = a\(Y'X) (avoids a large interims matrix)
    # 2) we permute Y c up front, the center term is symmetric, so we get cY'b c b' X
    # and (b'X) again avoids a large interims matrix, so does Y'b.
    return tr(a \ (Y' * X)) - (1 / 2) * tr(a \ ((Y' * b) * (a \ (b' * X))))
end

@doc raw"""
    inv(::SymplecticStiefel, A)
    inv!(::SymplecticStiefel, q, p)

Compute the symplectic inverse ``A^+`` of matrix ``A ∈ ℝ^{2n×2k}``.
Given a matrix
````math
A ∈ ℝ^{2n×2k},\quad
A =
\begin{bmatrix}
A_{1, 1} & A_{1, 2} \\
A_{2, 1} & A_{2, 2}
\end{bmatrix}, \quad A_{i, j} ∈ ℝ^{2n×2k}
````

the symplectic inverse is defined as:

```math
A^{+} := J_{2k}^{\mathrm{T}} A^{\mathrm{T}} J_{2n},
```

where ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).

The symplectic inverse of a matrix A can be expressed explicitly as:
```math
A^{+} =
  \begin{bmatrix}
    A_{2, 2}^{\mathrm{T}} & -A_{1, 2}^{\mathrm{T}} \\[1.2mm]
   -A_{2, 1}^{\mathrm{T}} &  A_{1, 1}^{\mathrm{T}}
  \end{bmatrix}.
```
"""
function Base.inv(M::SymplecticStiefel, p)
    q = similar(p')
    return inv!(M, q, p)
end

function inv!(M::SymplecticStiefel, q, p)
    n, k = get_parameter(M.size)
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

Compute the Cayley Inverse Retraction ``X = \mathcal{L}_p^{\mathrm{SpSt}}(q)``
such that the Cayley Retraction from ``p`` along ``X`` lands at ``q``, i.e.
``\mathcal{R}_p(X) = q`` [BendokatZimmermann:2021](@cite).

For ``p, q ∈ \mathrm{SpSt}(2n, 2k, ℝ)`` we can define the
inverse cayley retraction as long as the following matrices exist.
````math
    U = (I + p^+ q)^{-1} ∈ ℝ^{2k×2k},
    \quad
    V = (I + q^+ p)^{-1} ∈ ℝ^{2k×2k},
````

where ``(⋅)^+`` denotes the [`symplectic_inverse`](@ref).

THen the inverse retraction reads
````math
\mathcal{L}_p^{\mathrm{Sp}}(q) = 2p\bigl(V - U\bigr) + 2\bigl((p + q)U - p\bigr) ∈ T_p\mathrm{Sp}(2n).
````
"""
inverse_retract(::SymplecticStiefel, p, q, ::CayleyInverseRetraction)

function inverse_retract_cayley!(M::SymplecticStiefel, X, p, q)
    U_inv = lu!(add_scaled_I!(symplectic_inverse_times(M, p, q), 1))
    V_inv = lu!(add_scaled_I!(symplectic_inverse_times(M, q, p), 1))

    X .= 2 .* ((p / V_inv .- p / U_inv) .+ ((p + q) / U_inv) .- p)
    return X
end

"""
    is_flat(::SymplecticStiefel)

Return false. [`SymplecticStiefel`](@ref) is not a flat manifold.
"""
is_flat(M::SymplecticStiefel) = false

@doc raw"""
    manifold_dimension(::SymplecticStiefel)

Returns the dimension of the symplectic Stiefel manifold embedded in ``ℝ^{2n×2k}``,
i.e. [BendokatZimmermann:2021](@cite)
````math
    \operatorname{dim}(\mathrm{SpSt}(2n, 2k)) = (4n - 2k + 1)k.
````
"""
function manifold_dimension(M::SymplecticStiefel)
    n, k = get_parameter(M.size)
    return (4n - 2k + 1) * k
end

@doc raw"""
    project(::SymplecticStiefel, p, A)
    project!(::SymplecticStiefel, Y, p, A)

Given a point ``p ∈ \mathrm{SpSt}(2n, 2k)``,
project an element ``A ∈ ℝ^{2n×2k}`` onto
the tangent space ``T_p\mathrm{SpSt}(2n, 2k)`` relative to
the euclidean metric of the embedding ``ℝ^{2n×2k}``.

That is, we find the element ``X ∈ T_p\mathrm{SpSt}(2n, 2k)``
which solves the constrained optimization problem

````math
    \displaystyle\operatorname{min}_{X ∈ ℝ^{2n×2k}} \frac{1}{2}||X - A||^2, \quad
    \text{s.t.}\;
    h(X) := X^{\mathrm{T}} J p + p^{\mathrm{T}} J X = 0,
````
where ``h : ℝ^{2n×2k} → \operatorname{skew}(2k)`` defines
the restriction of ``X`` onto the tangent space ``T_p\mathrm{SpSt}(2n, 2k)``.
"""
project(::SymplecticStiefel, p, A)

function project!(::SymplecticStiefel, Y, p, A)
    J = SymplecticElement(Y, p, A)
    Jp = J * p

    function h(X)
        XTJp = X' * Jp
        return XTJp .- XTJp'
    end

    # Solve for Λ (Lagrange mutliplier):
    pT_p = p' * p  # (2k×2k)
    Λ = sylvester(pT_p, pT_p, h(A) ./ 2)

    Y[:, :] = A .- Jp * (Λ .- Λ')
    return Y
end

@doc raw"""
    rand(M::SymplecticStiefel; vector_at=nothing, σ = 1.0)

Generate a random point ``p ∈ \mathrm{SpSt}(2n, 2k)`` or
a random tangent vector ``X ∈ T_p\mathrm{SpSt}(2n, 2k)``
if `vector_at` is set to a point ``p ∈ \mathrm{Sp}(2n)``.

A random point on ``\mathrm{SpSt}(2n, 2k)`` is found by first generating a
random point on the symplectic manifold ``\mathrm{Sp}(2n)``,
and then projecting onto the Symplectic Stiefel manifold using the
[`canonical_project`](@ref) ``π_{\mathrm{SpSt}(2n, 2k)}``.
That is, ``p = π_{\mathrm{SpSt}(2n, 2k)}(p_{\mathrm{Sp}})``.

To generate a random tangent vector in ``T_p\mathrm{SpSt}(2n, 2k)``
this code exploits the second tangent vector space parametrization of
[`SymplecticStiefel`](@ref), that any ``X ∈ T_p\mathrm{SpSt}(2n, 2k)``
can be written as ``X = pΩ_X + p^sB_X``.
To generate random tangent vectors at ``p`` then, this function sets ``B_X = 0``
and generates a random Hamiltonian matrix ``Ω_X ∈ \mathfrak{sp}(2n,F)`` with
Frobenius norm of `σ` before returning ``X = pΩ_X``.
"""
rand(M::SymplecticStiefel; σ::Real=1.0, kwargs...)

function Random.rand!(
    rng::AbstractRNG,
    M::SymplecticStiefel,
    pX;
    vector_at=nothing,
    σ::Real=1.0,
)
    n, k = get_parameter(M.size)
    if vector_at === nothing
        canonical_project!(M, pX, rand(rng, SymplecticMatrices(2n); σ=σ))
        return pX
    else
        return random_vector!(rng, M, pX, vector_at; σ=σ)
    end
end

function random_vector!(rng, M::SymplecticStiefel, X, p; σ=1.0)
    k = get_parameter(M.size)[2]
    Ω = @view(X[1:(2k), 1:(2k)]) # use this memory
    rand!(rng, HamiltonianMatrices(2k), Ω; σ=σ)
    X .= p * Ω
    return X
end

@doc raw"""
    retract(::SymplecticStiefel, p, X, ::CayleyRetraction)
    retract!(::SymplecticStiefel, q, p, X, ::CayleyRetraction)

Compute the Cayley retraction on the Symplectic Stiefel manifold,
from `p` along `X` (computed inplace of `q`).

Given a point ``p ∈ \mathrm{SpSt}(2n, 2k)``, every tangent vector
``X ∈ T_p\mathrm{SpSt}(2n, 2k)`` is of the form
``X = \tilde{\Omega}p``, with
````math
    \tilde{\Omega} = \left(I_{2n} - \frac{1}{2}pp^+\right)Xp^+ -
                     pX^+\left(I_{2n} - \frac{1}{2}pp^+\right) ∈ ℝ^{2n×2n},
````
as shown in Proposition 3.5 of [BendokatZimmermann:2021](@cite).
Using this representation of ``X``, the Cayley retraction
on ``\mathrm{SpSt}(2n, 2k)`` is defined pointwise as
````math
    \mathcal{R}_p(X) = \operatorname{cay}\left(\frac{1}{2}\tilde{\Omega}\right)p.
````
The operator ``\operatorname{cay}(A) = (I - A)^{-1}(I + A)`` is the Cayley transform.

However, the computation of an ``2n×2n`` matrix inverse in the expression
above can be reduced down to inverting a ``2k×2k`` matrix due to Proposition
5.2 of [BendokatZimmermann:2021](@cite).

Let ``A = p^+X`` and ``H = X - pA``. Then an equivalent expression for the Cayley
retraction defined pointwise above is

```math
  \mathcal{R}_p(X) = -p + (H + 2p)(H^+H/4 - A/2 + I_{2k})^{-1}.
```

This expression is computed inplace of `q`.
"""
retract(::SymplecticStiefel, p, X, ::CayleyRetraction)

function retract_cayley!(M::SymplecticStiefel, q, p, X, t::Number)
    tX = t * X
    # Define intermediate matrices for later use:
    A = symplectic_inverse_times(M, p, tX)

    H = tX .- p * A  # Allocates (2n×2k).

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
    X = riemannian_gradient(::SymplecticStiefel, f, p, Y; embedding_metric::EuclideanMetric=EuclideanMetric())
    riemannian_gradient!(::SymplecticStiefel, f, X, p, Y; embedding_metric::EuclideanMetric=EuclideanMetric())

Compute the riemannian gradient `X` of `f` on [`SymplecticStiefel`](@ref)  at a point `p`,
provided that the gradient of the function ``\tilde f``, which is `f` continued into the embedding
is given by `Y`. The metric in the embedding is the Euclidean metric.

The manifold gradient `X` is computed from `Y` as

```math
    X = Yp^{\mathrm{T}}p + J_{2n}pY^{\mathrm{T}}J_{2n}p,
```

where ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).


"""
function riemannian_gradient(::SymplecticStiefel, p, Y)
    Jp = SymplecticElement(p, Y) * p
    return Y * (p' * p) .+ Jp * (Y' * Jp)
end

function riemannian_gradient!(
    ::SymplecticStiefel,
    X,
    p,
    Y;
    embedding_metric::EuclideanMetric=EuclideanMetric(),
)
    Jp = SymplecticElement(p, Y) * p
    X .= Y * (p' * p) .+ Jp * (Y' * Jp)
    return X
end

function Base.show(io::IO, ::SymplecticStiefel{TypeParameter{Tuple{n,k}},𝔽}) where {n,k,𝔽}
    return print(io, "SymplecticStiefel($(2n), $(2k); field=$(𝔽))")
end
function Base.show(io::IO, M::SymplecticStiefel{Tuple{Int,Int},𝔽}) where {𝔽}
    n, k = get_parameter(M.size)
    return print(io, "SymplecticStiefel($(2n), $(2k); field=$(𝔽); parameter=:field)")
end

@doc raw"""
    symplectic_inverse_times(::SymplecticStiefel, p, q)
    symplectic_inverse_times!(::SymplecticStiefel, A, p, q)

Directly compute the symplectic inverse of ``p ∈ \mathrm{SpSt}(2n, 2k)``,
multiplied with ``q ∈ \mathrm{SpSt}(2n, 2k)``.
That is, this function efficiently computes
``p^+q = (J_{2k}p^{\mathrm{T}}J_{2n})q ∈ ℝ^{2k×2k}``,
where ``J_{2n}, J_{2k}`` are the [`SymplecticElement`](@ref)
of sizes ``2n×2n`` and ``2k×2k`` respectively.

This function performs this common operation without allocating more than
a ``2k×2k`` matrix to store the result in, or in the case of the in-place
function, without allocating memory at all.
"""
function symplectic_inverse_times(M::SymplecticStiefel, p, q)
    n, k = get_parameter(M.size)
    A = similar(p, (2k, 2k))
    return symplectic_inverse_times!(M, A, p, q)
end

function symplectic_inverse_times!(M::SymplecticStiefel, A, p, q)
    n, k = get_parameter(M.size)
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
