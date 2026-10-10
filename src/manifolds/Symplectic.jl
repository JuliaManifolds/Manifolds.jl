@doc raw"""
    SymplecticMatrices{𝔽, T} <: AbstractDecoratorManifold{𝔽}

The symplectic manifold consists of all ``2n×2n`` matrices which preserve
the canonical symplectic form over ``𝔽^{2n×2n}×𝔽^{2n×2n}`` given by

```math
  \omega\colon 𝔽^{2n×2n}×𝔽^{2n×2n} → 𝔽,
  \quad \omega(x, y) = p^{\mathrm{T}} J_{2n} q, \  x, y \in 𝔽^{2n×2n},
```

where ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).

The symplectic manifold consists of

```math
\mathrm{Sp}(2n, ℝ) = \bigl\{ p ∈ ℝ^{2n×2n} \, \big| \, p^{\mathrm{T}}J_{2n}p = J_{2n} \bigr\},
```

The tangent space at a point ``p`` is given by [BendokatZimmermann:2021](@cite)

```math
\begin{align*}
  T_p\mathrm{Sp}(2n)
    &= \{X \in ℝ^{2n×2n} \ |\ p^{T}J_{2n}X + X^{T}J_{2n}p = 0 \}, \\
    &= \{X = pJ_{2n}S \ \mid\ S ∈ R^{2n×2n}, S^{\mathrm{T}} = S \}.
\end{align*}
```

# Constructor

    SymplecticMatrices(2n, field=ℝ; parameter::Symbol=:type)

Generate the (real-valued) symplectic manifold of ``2n×2n`` symplectic matrices.
The constructor for the [`SymplecticMatrices`](@ref) manifold accepts the even column/row embedding
dimension ``2n`` for the real symplectic manifold, ``ℝ^{2n×2n}``.
"""
struct SymplecticMatrices{𝔽, T} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function SymplecticMatrices(two_n::Int, field::AbstractNumbers = ℝ; parameter::Symbol = :type)
    two_n % 2 == 0 || throw(
        ArgumentError(
            "The matrix size `2n` of the symplectic manifold must be even, but was $(two_n).",
        ),
    )
    size = wrap_type_parameter(parameter, (div(two_n, 2),))
    return SymplecticMatrices{field, typeof(size)}(size)
end

@doc raw"""
    RealSymplecticMetric <: RiemannianMetric

The canonical Riemannian metric on the symplectic manifold,
defined pointwise for ``p \in \mathrm{Sp}(2n)`` by [Fiori:2011](@cite)]

```math
\begin{align*}
  & g_p \colon T_p\mathrm{Sp}(2n)×T_p\mathrm{Sp}(2n) → ℝ, \\
  & g_p(Z_1, Z_2) = \operatorname{tr}((p^{-1}Z_1)^{\mathrm{T}} (p^{-1}Z_2)).
\end{align*}
```

This metric is also the default metric for the [`SymplecticMatrices`](@ref) manifold.
"""
struct RealSymplecticMetric <: RiemannianMetric end

@doc raw"""
    ExtendedSymplecticMetric <: AbstractMetric

The extension of the [`RealSymplecticMetric`](@ref) at a point ``p \in \mathrm{Sp}(2n)``
as an inner product over the embedding space ``ℝ^{2n×2n}``, i.e.

```math
    ⟨x, y⟩_p = ⟨p^{-1}x, p^{-1}⟩_{\mathrm{Fr}}
    = \operatorname{tr}(x^{\mathrm{T}}(pp^{\mathrm{T}})^{-1}y), \text{ for all } x, y \in ℝ^{2n×2n}.
```
"""
struct ExtendedSymplecticMetric <: AbstractMetric end

@doc raw"""
    SymplecticElement{T}

A lightweight structure to represent the action of the matrix
representation of the canonical symplectic form,

```math
J_{2n}(λ) = λ\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n
\end{bmatrix} ∈ ℝ^{2n×2n},
```

where we write ``J_{2n} = J_{2n}(1)`` for short.
The canonical symplectic form is represented by

```math
\omega_{2n}(x, y) = x^{\mathrm{T}}J_{2n}y, \quad x, y ∈ ℝ^{2n}.
```

The entire matrix is however not instantiated in memory, instead a scalar
``λ`` of type `T` is stored, which is used to keep track of scaling and transpose operations
applied  to each `SymplecticElement`.
This type acts similar to `I` from `LinearAlgeba`.

# Constructor

    SymplecticElement(λ=1)

Generate the sumplectic matrix with scaling ``1``.
"""
struct SymplecticElement{T}
    λ::T
end
SymplecticElement() = SymplecticElement(1)
SymplecticElement(λ::T) where {T <: Number} = SymplecticElement{T}(λ)

function SymplecticElement(arrays::Vararg{AbstractArray})
    TS = Base.promote_type(map(eltype, arrays)...)
    return SymplecticElement(one(TS))
end

@doc raw"""
    change_representer(::SymplecticMatrices, ::EuclideanMetric, p, X)
    change_representer!(::SymplecticMatrices, Y, ::EuclideanMetric, p, X)

Compute the representation of a tangent vector ``ξ ∈ T_p\mathrm{Sp}(2n, ℝ)`` s.t.
```math
  g_p(c_p(ξ), η) = ⟨ξ, η⟩^{\text{Euc}} \text{for all } η ∈ T_p\mathrm{Sp}(2n, ℝ).
```
with the conversion function

```math
  c_p : T_p\mathrm{Sp}(2n, ℝ) → T_p\mathrm{Sp}(2n, ℝ), \quad
  c_p(ξ) = \frac{1}{2} pp^{\mathrm{T}} ξ + \frac{1}{2} pJ_{2n} ξ^{\mathrm{T}} pJ_{2n},
```

where ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).

Each of the terms ``c_p^1(ξ) = p p^{\mathrm{T}} ξ`` and ``c_p^2(ξ) = pJ_{2n} ξ^{\mathrm{T}} pJ_{2n}`` from the
above definition of ``c_p(η)`` are themselves metric compatible in the sense that

```math
    c_p^i : T_p\mathrm{Sp}(2n, ℝ) → ℝ^{2n×2n}\quad
    g_p^i(c_p(ξ), η) = ⟨ξ, η⟩^{\text{Euc}} \;∀\; η ∈ T_p\mathrm{Sp}(2n, ℝ),
```

for ``i \in {1, 2}``. However the range of each function alone is not confined to
  ``T_p\mathrm{Sp}(2n, ℝ)``, but the convex combination

```math
    c_p(ξ) = \frac{1}{2}c_p^1(ξ) + \frac{1}{2}c_p^2(ξ)
```

does have the correct range ``T_p\mathrm{Sp}(2n, ℝ)``.
"""
change_representer(::SymplecticMatrices, ::EuclideanMetric, p, X)

function change_representer!(::SymplecticMatrices, Y, ::EuclideanMetric, p, X)
    J = SymplecticElement(p, X) # J_{2n}
    pT_X = p' * X
    Y .= (1 / 2) .* p * (pT_X .+ J * pT_X' * J)
    return Y
end

@doc raw"""
    change_representer(MetMan::MetricManifold{<:Any, <:Euclidean, ExtendedSymplecticMetric},
                       EucMet::EuclideanMetric, p, X)
    change_representer!(MetMan::MetricManifold{<:Any, <:Euclidean, ExtendedSymplecticMetric},
                        Y, EucMet::EuclideanMetric, p, X)

Change the representation of a matrix ``ξ ∈ ℝ^{2n×2n}``
into the inner product space ``(ℝ^{2n×2n}, g_p)`` where the inner product
is given by
``g_p(ξ, η) = \langle p^{-1}ξ, p^{-1}η \rangle = \operatorname{tr}(ξ^{\mathrm{T}}(pp^{\mathrm{T}})^{-1}η)``,
as the extension of the [`RealSymplecticMetric`](@ref) onto the entire embedding space.

By changing the representation we mean to apply a mapping
````math
    c_p : ℝ^{2n×2n} → ℝ^{2n×2n},
````
defined by requiring that it satisfy the metric compatibility condition
````math
    g_p(c_p(ξ), η) = ⟨p^{-1}c_p(ξ), p^{-1}η⟩ = ⟨ξ, η⟩^{\text{Euc}}
        \;∀\; η ∈ T_p\mathrm{Sp}(2n, ℝ).
````
In this case, we compute the mapping
````math
    c_p(ξ) = pp^{\mathrm{T}} ξ.
````
"""
function change_representer(
        ::MetricManifold{<:Any, <:Euclidean, ExtendedSymplecticMetric},
        ::EuclideanMetric,
        p,
        X,
    )
    return p * p' * X
end

function change_representer!(
        ::MetricManifold{<:Any, <:Euclidean, ExtendedSymplecticMetric},
        Y,
        ::EuclideanMetric,
        p,
        X,
    )
    Y .= p * p' * X
    return Y
end

@doc raw"""
    check_point(M::SymplecticMatrices, p; kwargs...)

Check whether `p` is a valid point on the [`SymplecticMatrices`](@ref) `M`=$\mathrm{Sp}(2n)$,
i.e. that it has the right [`AbstractNumbers`](@extref ManifoldsBase number-system) type and ``p^{+}p`` is (approximately)
the identity, where ``A^+`` denotes the [`symplectic_inverse`](@ref).

The tolerance can be set with `kwargs...`.
"""
function check_point(
        M::SymplecticMatrices,
        p::T;
        atol::Real = sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
        kwargs...,
    ) where {T}
    # Perform check that the matrix lives on the real symplectic manifold:
    if !isapprox(inv(M, p) * p, LinearAlgebra.I; atol = atol, kwargs...)
        return DomainError(
            norm(inv(M, p) * p - LinearAlgebra.I),
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

Checks whether `X` is a valid tangent vector at `p` on the [`SymplecticMatrices`](@ref)
`M`=``\mathrm{Sp}(2n)``, which requires that

```math
p^{T}J_{2n}X + X^{T}J_{2n}p = 0
```
holds (approximately), where ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).

The tolerance can be set with `kwargs...`
"""
check_vector(::SymplecticMatrices, ::Any...)

function check_vector(M::SymplecticMatrices, p, X::T; kwargs...) where {T}
    J = SymplecticElement(p, X)
    if !isapprox(X' * J * p, -p' * J * X; kwargs...)
        return DomainError(
            norm(X' * J * p + p' * J * X, 2),
            (
                "The matrix X is not in the tangent space at point p of the" *
                    " manifold $(M), as X'Jp + p'JX is not the zero matrix."
            ),
        )
    end
    return nothing
end

function ManifoldsBase.default_inverse_retraction_method(::SymplecticMatrices)
    return CayleyInverseRetraction()
end

metric(::SymplecticMatrices) = RealSymplecticMetric()

ManifoldsBase.default_retraction_method(::SymplecticMatrices) = CayleyRetraction()

@doc raw"""
    distance(M::SymplecticMatrices, p, q)

Compute an approximate geodesic distance between two Symplectic matrices
``p, q \in \mathrm{Sp}(2n)``, as done in [WangSunFiori:2018](@cite).

````math
  \operatorname{dist}(p, q)
    ≈ \lVert\operatorname{Log}(p^+q)\rVert_{\mathrm{Fr}},
````
where the ``\operatorname{Log}(⋅)`` operator is the matrix logarithm.

This approximation is justified by first recalling the Baker-Campbell-Hausdorf formula,

````math
\operatorname{Log}(\operatorname{Exp}(A)\operatorname{Exp}(B))
 = A + B + \frac{1}{2}[A, B] + \frac{1}{12}[A, [A, B]] + \frac{1}{12}[B, [B, A]]
    + \ldots \;.
````

Then we write the expression for the exponential map from ``p`` to ``q`` as

````math
    q =
    \operatorname{exp}_p(X)
    =
    p \operatorname{Exp}((p^{+}X)^{\mathrm{T}})
    \operatorname{Exp}([p^{+}X - (p^{+}X)^{\mathrm{T}}]),
    X \in T_p\mathrm{Sp},
````

and with the geodesic distance between ``p`` and ``q`` given by

```math
\operatorname{dist}(p, q) = \lVert X \rVert_p = \lVert p^+ X \rVert_{\mathrm{Fr}}
```

we see that

```math
  \begin{align*}
   \lVert\operatorname{Log}(p^+q)\rVert_{\mathrm{Fr}}
    &=\Bigl\lVert
        \operatorname{Log}\bigl(
            \operatorname{Exp}((p^{+}X)^{\mathrm{T}})
            \operatorname{Exp}(p^{+}X - (p^{+}X)^{\mathrm{T}})
        \bigr)
    \Bigr\rVert_{\mathrm{Fr}} \\
    &=\lVert p^{+}X + \frac{1}{2}[(p^{+}X)^{\mathrm{T}}, p^{+}X - (p^{+}X)^{\mathrm{T}}]
        + \ldots\lVert_{\mathrm{Fr}} \\
    &≈\lVert p^{+}X\rVert_{\mathrm{Fr}} = \operatorname{dist}(p, q).
  \end{align*}
```
"""
function distance(M::SymplecticMatrices, p, q)
    return norm(log(symplectic_inverse_times(M, p, q)))
end

embed(::SymplecticMatrices, p) = p
embed(::SymplecticMatrices, p, X) = X

@doc raw"""
    exp(M::SymplecticMatrices, p, X)
    exp!(M::SymplecticMatrices, q, p, X)

The Exponential mapping on the Symplectic manifold with the
[`RealSymplecticMetric`](@ref) Riemannian metric.

For the point ``p \in \mathrm{Sp}(2n)`` the exponential mapping along the tangent
vector ``X \in T_p\mathrm{Sp}(2n)`` is computed as [WangSunFiori:2018](@cite)
````math
    \operatorname{exp}_p(X) = p \operatorname{Exp}((p^{-1}X)^{\mathrm{T}})
                                \operatorname{Exp}(p^{-1}X - (p^{-1}X)^{\mathrm{T}}),
````
where ``\operatorname{Exp}(⋅)`` denotes the matrix exponential.
"""
exp(::SymplecticMatrices, ::Any, ::Any)

function exp!(M::SymplecticMatrices, q, p, X)
    p_star_X = symplectic_inverse_times(M, p, X)
    q .= p * exp(Array(p_star_X')) * exp(p_star_X - p_star_X')
    return q
end

function get_embedding(::SymplecticMatrices{𝔽, TypeParameter{Tuple{n}}}) where {n, 𝔽}
    return Euclidean(2 * n, 2 * n; field = 𝔽)
end
function get_embedding(M::SymplecticMatrices{𝔽, Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return Euclidean(2 * n, 2 * n; field = 𝔽, parameter = :field)
end

function ManifoldsBase.get_embedding_type(::SymplecticMatrices)
    return ManifoldsBase.EmbeddedManifoldType()
end

@doc raw"""
    gradient(M::SymplecticMatrices, f, p, backend::RiemannianProjectionBackend; extended_metric=true)
    gradient!(M::SymplecticMatrices, f, p, backend::RiemannianProjectionBackend; extended_metric=true)

Compute the manifold gradient ``\text{grad}f(p)`` of a scalar function
``f \colon \mathrm{Sp}(2n) → ℝ`` at
``p \in \mathrm{Sp}(2n)``.

The element ``\text{grad}f(p)`` is found as the Riesz representer of the differential
``\text{D}f(p) \colon T_p\mathrm{Sp}(2n) → ℝ`` with respect to
the Riemannian metric inner product at ``p`` [Fiori:2011](@cite)].
That is, ``\text{grad}f(p) \in T_p\mathrm{Sp}(2n)`` solves the relation

```math
    g_p(\text{grad}f(p), X) = \text{D}f(p) \quad\forall\; X \in T_p\mathrm{Sp}(2n).
```

The default behaviour is to first change the representation of the Euclidean gradient from
the Euclidean metric to the [`RealSymplecticMetric`](@ref) at ``p``, and then we projecting
the result onto the correct tangent tangent space ``T_p\mathrm{Sp}(2n, ℝ)``
w.r.t the Riemannian metric ``g_p`` extended to the entire embedding space.

# Arguments:
* `extended_metric = true`: If `true`, compute the gradient ``\text{grad}f(p)`` by
    first changing the representer of the Euclidean gradient of a smooth extension
    of ``f``, ``∇f(p)``, with respect to the [`RealSymplecticMetric`](@ref) at ``p``
    extended to the entire embedding space, before projecting onto the correct
    tangent vector space with respect to the same extended metric ``g_p``.
    If `false`, compute the gradient by first projecting ``∇f(p)`` onto the
    tangent vector space, before changing the representer in the tangent
    vector space to comply with the [`RealSymplecticMetric`](@ref).
"""
function ManifoldDiff.gradient(
        M::SymplecticMatrices,
        f,
        p,
        backend::RiemannianProjectionBackend;
        extended_metric = true,
    )
    Y = allocate_result(M, gradient, p)
    return gradient!(M, f, Y, p, backend; extended_metric = extended_metric)
end

function ManifoldDiff.gradient!(
        M::SymplecticMatrices,
        f,
        X,
        p,
        backend::RiemannianProjectionBackend;
        extended_metric = true,
    )
    _gradient!(f, X, p, backend.diff_backend)
    if extended_metric
        MetricM = MetricManifold(get_embedding(M), ExtendedSymplecticMetric())
        change_representer!(MetricM, X, EuclideanMetric(), p, X)
        return project!(MetricM, X, p, X)
    else
        project!(M, X, p, X)
        return change_representer!(M, X, EuclideanMetric(), p, X)
    end
end

@doc raw"""
    inner(::SymplecticMatrices{ℝ}, p, X, Y)

Compute the canonical Riemannian inner product [`RealSymplecticMetric`](@ref)
````math
    g_p(X, Y) = \operatorname{tr}((p^{-1}X)^{\mathrm{T}} (p^{-1}Y))
````
between the two tangent vectors ``X, Y \in T_p\mathrm{Sp}(2n)``.
"""
function inner(M::SymplecticMatrices{ℝ}, p, X, Y)
    p_star = inv(M, p)
    return dot((p_star * X), (p_star * Y))
end

@doc raw"""
    symplectic_inverse(A)

Given a matrix
```math
  A ∈ ℝ^{2n×2k},\quad
  A =
  \begin{bmatrix}
  A_{1,1} & A_{1,2} \\
  A_{2,1} & A_{2, 2}
  \end{bmatrix}
```

the symplectic inverse is defined as:

```math
A^{+} := J_{2k}^{\mathrm{T}} A^{\mathrm{T}} J_{2n},
```

where ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).

The symplectic inverse of A can be expressed explicitly as:

```math
A^{+} =
  \begin{bmatrix}
    A_{2, 2}^{\mathrm{T}} & -A_{1, 2}^{\mathrm{T}} \\[1.2mm]
   -A_{2, 1}^{\mathrm{T}} &  A_{1, 1}^{\mathrm{T}}
  \end{bmatrix}.
```
"""
function symplectic_inverse(A::AbstractMatrix)
    N, K = size(A)
    @assert iseven(N) "The first matrix dimension of A ($N) has to be even"
    @assert iseven(K) "The second matrix dimension of A ($K) has to be even"
    n = div(N, 2)
    k = div(K, 2)
    Ai = similar(A')
    checkbounds(A, 1:(2n), 1:(2k))
    @inbounds for i in 1:k, j in 1:n
        Ai[i, j] = A[j + n, i + k]
    end
    @inbounds for i in 1:k, j in 1:n
        Ai[i + k, j] = -A[j + n, i]
    end
    @inbounds for i in 1:k, j in 1:n
        Ai[i, j + n] = -A[j, i + k]
    end
    @inbounds for i in 1:k, j in 1:n
        Ai[i + k, j + n] = A[j, i]
    end
    return Ai
end

@doc raw"""
    inv(::SymplecticMatrices, A)
    inv!(::SymplecticMatrices, A)

Compute the symplectic inverse ``A^+`` of matrix ``A ∈ ℝ^{2n×2n}``.
See [`symplectic_inverse`](@ref) for details.

"""
function Base.inv(::SymplecticMatrices{ℝ}, A)
    return symplectic_inverse(A)
end

function symplectic_inverse!(A)
    n = div(size(A, 1), 2)
    checkbounds(A, 1:(2n), 1:(2n))
    @inbounds for i in 1:n, j in 1:n
        A[i, j], A[j + n, i + n] = A[j + n, i + n], A[i, j]
    end
    @inbounds for i in 1:n, j in i:n
        if i == j
            A[i, j + n] = -A[i, j + n]
        else
            A[i, j + n], A[j, i + n] = -A[j, i + n], -A[i, j + n]
        end
    end
    @inbounds for i in 1:n, j in i:n
        if i == j
            A[i + n, j] = -A[i + n, j]
        else
            A[i + n, j], A[j + n, i] = -A[j + n, i], -A[i + n, j]
        end
    end
    return A
end

@doc raw"""
    inv!(M::SymplecticMatrices, A)

Compute the [`symplectic_inverse`](@ref) of a square matrix A in place of A
"""
function inv!(M::SymplecticMatrices{ℝ}, A)
    return symplectic_inverse!(A)
end

@doc raw"""
    inverse_retract(M::SymplecticMatrices, p, q, ::CayleyInverseRetraction)

Compute the Cayley Inverse Retraction ``X = \mathcal{L}_p^{\mathrm{Sp}}(q)``
such that the Cayley Retraction from ``p`` along ``X`` lands at ``q``, i.e.
``\mathcal{R}_p(X) = q`` [BendokatZimmermann:2021](@cite).

For ``p, q ∈ \mathrm{Sp}(2n, ℝ)`` then, we can define the
inverse Cayley retraction as long as the following matrices exist.
````math
    U = (I + p^+ q)^{-1}, \quad V = (I + q^+ p)^{-1},
````

where ``(⋅)^+`` denotes the [`symplectic_inverse`](@ref).

Then inverse Cayley retraction at ``p`` applied to ``q`` is
```math
\mathcal{L}_p^{\mathrm{Sp}}(q)
  = 2p\bigl(V - U\bigr) + 2\bigl((p + q)U - p\bigr) ∈ T_p\mathrm{Sp}(2n).
```
"""
inverse_retract(::SymplecticMatrices, p, q, ::CayleyInverseRetraction)

function inverse_retract_cayley!(M::SymplecticMatrices, X, p, q)
    U_inv = lu(add_scaled_I!(symplectic_inverse_times(M, p, q), 1))
    V_inv = lu(add_scaled_I!(symplectic_inverse_times(M, q, p), 1))

    X .= 2 .* ((p / V_inv .- p / U_inv) .+ ((p + q) / U_inv) .- p)
    return X
end

"""
    is_flat(::SymplecticMatrices)

Return false. [`SymplecticMatrices`](@ref) is not a flat manifold.
"""
is_flat(M::SymplecticMatrices) = false

@doc raw"""
    manifold_dimension(::SymplecticMatrices)

Returns the dimension of the symplectic manifold
embedded in ``ℝ^{2n×2n}``, i.e.
```math
  \operatorname{dim}(\mathrm{Sp}(2n)) = (2n + 1)n.
```
"""
function manifold_dimension(M::SymplecticMatrices)
    n = get_parameter(M.size)[1]
    return (2n + 1) * n
end

@doc raw"""
    project(::SymplecticMatrices, p, A)
    project!(::SymplecticMatrices, Y, p, A)

Given a point ``p \in \mathrm{Sp}(2n)``,
project an element ``A \in ℝ^{2n×2n}`` onto
the tangent space ``T_p\mathrm{Sp}(2n)`` relative to
the euclidean metric of the embedding ``ℝ^{2n×2n}``.

That is, we find the element ``X \in T_p\operatorname{Sp}(2n)``
which solves the constrained optimization problem
````math
    \operatorname{min}_{X \in ℝ^{2n×2n}} \frac{1}{2}\lVert X - A\rVert^2, \quad
    \text{such that}\;
    h(X) := X^{\mathrm{T}} J_{2n} p + p^{\mathrm{T}} J_{2n} X = 0,
````
where ``h: ℝ^{2n×2n} → \operatorname{skew}(2n)`` denotes
the restriction of ``X`` onto the tangent space ``T_p\operatorname{SpSt}(2n, 2k)``
and ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).
"""
project(::SymplecticMatrices, p, A)

function project!(::SymplecticMatrices, Y, p, A)
    J = SymplecticElement(Y, p, A)
    Jp = J * p

    function h(X)
        XtJp = X' * Jp
        return XtJp .- XtJp'
    end

    # Solve for Λ (Lagrange multiplier):
    pT_p = p' * p  # (2k×2k)
    Λ = sylvester(pT_p, pT_p, h(A) ./ 2)

    Y[:, :] = A .- Jp * (Λ .- Λ')
    return Y
end

@doc raw"""
    project!(::MetricManifold{𝔽,<:Euclidean,ExtendedSymplecticMetric}, Y, p, X) where {𝔽}

Compute the projection of ``X ∈ R^{2n×2n}`` onto ``T_p\mathrm{Sp}(2n, ℝ)`` with respect to
the [`RealSymplecticMetric`](@ref) ``g``.

The closed form projection mapping is given by [GaoSonAbsilStykel:2021](@cite)

````math
  \operatorname{P}^{T_p\mathrm{Sp}(2n)}_{g_p}(X) = pJ_{2n}\operatorname{sym}(p^{\mathrm{T}}J_{2n}^{\mathrm{T}}X),
````

where ``\operatorname{sym}(A) = \frac{1}{2}(A + A^{\mathrm{T}})`` and and ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).
"""
function project!(::MetricManifold{<:Any, <:Euclidean, ExtendedSymplecticMetric}, Y, p, X)
    J = SymplecticElement(p, X)

    pTJTX = p' * J' * X
    sym_pTJTX = (1 / 2) .* (pTJTX + pTJTX')
    Y .= p * J * (sym_pTJTX)
    return Y
end

@doc raw"""
    project_normal!(::MetricManifold{𝔽,<:Euclidean,ExtendedSymplecticMetric}, Y, p, X)

Project onto the normal of the tangent space ``(T_p\mathrm{Sp}(2n))^{\perp_g}`` at
a point ``p ∈ \mathrm{Sp}(2n)``, relative to the riemannian metric
``g`` [`RealSymplecticMetric`](@ref).

That is,

````math
(T_p\mathrm{Sp}(2n))^{\perp_g}
 = \{Y ∈ ℝ^{2n×2n} : g_p(Y, X) = 0 \test{ for all } X \in T_p\mathrm{Sp}(2n)\}.
````

The closed form projection operator onto the normal space is given by [GaoSonAbsilStykel:2021](@cite)

````math
\operatorname{P}^{(T_p\mathrm{Sp}(2n))\perp}_{g_p}(X) = pJ_{2n}\operatorname{skew}(p^{\mathrm{T}}J_{2n}^{\mathrm{T}}X),
````

where ``\operatorname{skew}(A) = \frac{1}{2}(A - A^{\mathrm{T}})``
and ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).

This function is not exported.
"""
function project_normal!(
        ::MetricManifold{𝔽, <:Euclidean, ExtendedSymplecticMetric},
        Y,
        p,
        X,
    ) where {𝔽}
    J = SymplecticElement(p, X)
    pTJTX = p' * J' * X
    skew_pTJTX = (1 / 2) .* (pTJTX .- pTJTX')
    Y .= p * J * skew_pTJTX
    return Y
end

@doc raw"""
    rand(::SymplecticMatrices; vector_at=nothing, σ::Real=1.0)

Generate a random point on ``\mathrm{Sp}(2n)`` or a random
tangent vector ``X \in T_p\mathrm{Sp}(2n)`` if `vector_at` is set to
a point ``p \in \mathrm{Sp}(2n)``.

A random point on ``\mathrm{Sp}(2n)`` is constructed by generating a
random Hamiltonian matrix ``Ω \in \mathfrak{sp}(2n,F)`` with norm `σ`,
and then transforming it to a symplectic matrix by applying the Cayley transform

```math
  \operatorname{cay}: \mathfrak{sp}(2n,F) → \mathrm{Sp}(2n),
  \ \Omega \mapsto (I - \Omega)^{-1}(I + \Omega).
```

To generate a random tangent vector in ``T_p\mathrm{Sp}(2n)``, this code employs the
second tangent vector space parametrization of [`SymplecticMatrices`](@ref).
It first generates a random symmetric matrix ``S`` by `S = randn(2n, 2n)`
and then symmetrizes it as `S = S + S'`.
Then ``S`` is normalized to have Frobenius norm of `σ`
and `X = pJS` is returned, where `J` is the [`SymplecticElement`](@ref).
"""
rand(SymplecticMatrices; σ::Real = 1.0, kwargs...)

function Random.rand!(
        rng::AbstractRNG,
        M::SymplecticMatrices,
        pX;
        vector_at = nothing,
        σ::Real = 1.0,
    )
    n = get_parameter(M.size)[1]
    if vector_at === nothing
        rand!(rng, HamiltonianMatrices(2n), pX; σ = σ)
        pX .= (I - pX) \ (I + pX)
        return pX
    else
        random_vector!(rng, M, pX, vector_at; σ = σ)
        return pX
    end
end

function random_vector!(rng, M::SymplecticMatrices, X, p; σ::Real = 1.0)
    n = get_parameter(M.size)[1]
    # Generate random symmetric matrix:
    randn!(rng, X)
    X .= 0.5 * (X + X')
    X .*= σ / norm(X)
    lmul!(SymplecticElement(p), X)
    X .= p * X
    return X
end

@doc raw"""
    retract(::SymplecticMatrices, p, X, ::CayleyRetraction)
    retract!(::SymplecticMatrices, q, p, X, ::CayleyRetraction)

Compute the Cayley retraction on ``p ∈ \mathrm{Sp}(2n, ℝ)`` in
the direction of tangent vector ``X ∈ T_p\mathrm{Sp}(2n, ℝ)``,
as defined in by Birtea et al in proposition 2 [BirteaCaşuComănescu:2020](@cite).

Using the [`symplectic_inverse`](@ref) ``A^+`` of a matrix ``A \in ℝ^{2n×2n}``
the retraction ``\mathcal{R}: T\mathrm{Sp}(2n) → \mathrm{Sp}(2n)``
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
"""
retract(M::SymplecticMatrices, p, X)

function ManifoldsBase.retract_cayley!(M::SymplecticMatrices, q, p, X)
    return ManifoldsBase.retract_cayley_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_cayley_fused!(M::SymplecticMatrices, q, p, X, t::Number)
    p_star_X = symplectic_inverse_times(M, p, t * X)

    divisor = lu(2 * I - p_star_X)
    q .= p * (divisor \ add_scaled_I!(p_star_X, 2.0))
    return q
end

@doc raw"""
    riemannian_gradient(M::SymplecticMatrices, p, Y)

Given a gradient ``Y = \operatorname{grad} \tilde f(p)`` in the embedding ``ℝ^{2n×2n}`` or at
least around the [`SymplecticMatrices`](@ref) `M` where `p` (the embedding of) a point on `M`,
we restrict ``\tilde f`` to the manifold and denote that by ``f``.
Then the Riemannian gradient ``X = \operatorname{grad} f(p)`` with respect to the
[`RealSymplecticMetric`](@ref), the left invariant metric of Section 2.3 in
[BirteaCaşuComănescu:2020](@cite), is given by Theorem 2 and Eq. (1) there as

```math
  X = \frac{1}{2}\bigl(pp^{\mathrm{T}}Y + pJ_{2n}Y^{\mathrm{T}}pJ_{2n}\bigr),
```

where ``J_{2n}`` denotes the [`SymplecticElement`](@ref).
"""
function riemannian_gradient(M::SymplecticMatrices, p, Y; kwargs...)
    return change_representer(M, EuclideanMetric(), p, Y)
end

function riemannian_gradient!(M::SymplecticMatrices, X, p, Y; kwargs...)
    return change_representer!(M, X, EuclideanMetric(), p, Y)
end

function Base.show(io::IO, ::SymplecticMatrices{𝔽, TypeParameter{Tuple{n}}}) where {n, 𝔽}
    return print(io, "SymplecticMatrices($(2n), $(𝔽))")
end
function Base.show(io::IO, M::SymplecticMatrices{𝔽, Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return print(io, "SymplecticMatrices($(2n), $(𝔽); parameter=:field)")
end

@doc raw"""
    symplectic_inverse_times(::SymplecticMatrices, p, q)
    symplectic_inverse_times!(::SymplecticMatrices, A, p, q)

Directly compute the symplectic inverse of ``p \in \mathrm{Sp}(2n)``,
multiplied with ``q \in \mathrm{Sp}(2n)``.
That is, this function efficiently computes
``p^+q = (J_{2n}p^{\mathrm{T}}J_{2n})q ∈ ℝ^{2n×2n}``,
where ``J_{2n} = \begin{bmatrix} 0_n & I_n \\ -I_n & 0_n \end{bmatrix}`` denotes the [`SymplecticElement`](@ref).

"""
function symplectic_inverse_times(M::SymplecticMatrices, p, q)
    A = similar(p)
    return symplectic_inverse_times!(M, A, p, q)
end

function symplectic_inverse_times!(M::SymplecticMatrices, A, p, q)
    n = get_parameter(M.size)[1]
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

ndims(J::SymplecticElement) = 2
copy(J::SymplecticElement) = SymplecticElement(copy(J.λ))
Base.eltype(::SymplecticElement{T}) where {T} = T
function Base.convert(::Type{SymplecticElement{T}}, J::SymplecticElement) where {T}
    return SymplecticElement(convert(T, J.λ))
end

function Base.show(io::IO, J::SymplecticElement)
    s = "$(J.λ)"
    if occursin(r"\w+\s*[\+\-]\s*\w+", s)
        s = "($s)"
    end
    return print(io, typeof(J), "(): $(s)*[0 I; -I 0]")
end

(Base.:-)(J::SymplecticElement) = SymplecticElement(-J.λ)

function (Base.:^)(J::SymplecticElement, n::Integer)
    return ifelse(
        n % 2 == 0,
        UniformScaling((-1)^(div(n, 2)) * (J.λ)^n),
        SymplecticElement((-1)^(div(n - 1, 2)) * (J.λ)^n),
    )
end

(Base.:*)(x::Number, J::SymplecticElement) = SymplecticElement(x * J.λ)
(Base.:*)(J::SymplecticElement, x::Number) = SymplecticElement(x * J.λ)
function (Base.:*)(J::SymplecticElement, K::SymplecticElement)
    return LinearAlgebra.UniformScaling(-J.λ * K.λ)
end

Base.transpose(J::SymplecticElement) = -J
Base.adjoint(J::SymplecticElement) = SymplecticElement(-conj(J.λ))
Base.inv(J::SymplecticElement) = SymplecticElement(-(1 / J.λ))

(Base.:+)(J::SymplecticElement, K::SymplecticElement) = SymplecticElement(J.λ + K.λ)
(Base.:-)(J::SymplecticElement, K::SymplecticElement) = SymplecticElement(J.λ - K.λ)

(Base.:+)(J::SymplecticElement, p::AbstractMatrix) = p + J
function (Base.:+)(p::AbstractMatrix, J::SymplecticElement)
    # When we are adding, the Matrices must match in size:
    two_n, two_k = size(p)
    if (two_n % 2 != 0) || (two_n != two_k)
        throw(
            ArgumentError(
                "'p' must be square with even row and dimension, " *
                    "was: ($(two_n), $(two_k)) != (2n, 2n).",
            ),
        )
    end
    n = div(two_n, 2)

    # Allocate new memory:
    TS = typeof(one(eltype(p)) + one(eltype(J)))
    out = copyto!(similar(p, TS), p)

    add_scaled_I!(view(out, 1:n, (n + 1):(2n)), J.λ)
    add_scaled_I!(view(out, (n + 1):(2n), 1:n), -J.λ)
    return out
end

# Binary minus:
(Base.:-)(J::SymplecticElement, p::AbstractMatrix) = J + (-p)
(Base.:-)(p::AbstractMatrix, J::SymplecticElement) = p + (-J)

function (Base.:*)(J::SymplecticElement, p::AbstractVecOrMat)
    two_n = size(p)[1]
    if two_n % 2 != 0
        throw(ArgumentError("'p' must have even row dimension, was: $(two_n) != 2n."))
    end
    n = div(two_n, 2)

    # Allocate new memory:
    TS = typeof(one(eltype(p)) + one(eltype(J)))
    Jp = similar(p, TS)

    # Perform left multiply by λ*J:
    mul!((@inbounds view(Jp, 1:n, :)), J.λ, @inbounds view(p, (n + 1):lastindex(p, 1), :))
    mul!((@inbounds view(Jp, (n + 1):lastindex(Jp, 1), :)), -J.λ, @inbounds view(p, 1:n, :))

    return Jp
end

function (Base.:*)(p::AbstractMatrix, J::SymplecticElement)
    two_k = size(p)[2]
    if two_k % 2 != 0
        throw(ArgumentError("'p' must have even column dimension, was: $(two_k) != 2k."))
    end
    k = div(two_k, 2)

    # Allocate new memory:
    TS = typeof(one(eltype(p)) + one(eltype(J)))
    pJ = similar(p, TS)

    # Perform right multiply by λ*J:
    mul!((@inbounds view(pJ, :, 1:k)), -J.λ, @inbounds view(p, :, (k + 1):lastindex(p, 2)))
    mul!((@inbounds view(pJ, :, (k + 1):lastindex(pJ, 2))), J.λ, @inbounds view(p, :, 1:k))
    return pJ
end

function LinearAlgebra.lmul!(J::SymplecticElement, p::AbstractVecOrMat)
    # Perform left multiplication by a symplectic matrix,
    # overwriting the matrix p in place:
    two_n = size(p)[1]
    if two_n % 2 != 0
        throw(ArgumentError("'p' must have even row dimension, was: $(two_n) != 2n."))
    end
    n = div(two_n, 2)

    half_row_p = copy(@inbounds view(p, 1:n, :))
    mul!((@inbounds view(p, 1:n, :)), J.λ, @inbounds view(p, (n + 1):lastindex(p, 1), :))
    mul!((@inbounds view(p, (n + 1):lastindex(p, 1), :)), -J.λ, @inbounds view(half_row_p, :, :))
    return p
end

function LinearAlgebra.rmul!(p::AbstractMatrix, J::SymplecticElement)
    # Perform right multiplication by a symplectic matrix,
    # overwriting the matrix p in place:
    two_k = size(p)[2]
    if two_k % 2 != 0
        throw(ArgumentError("'p' must have even column dimension, was: $(two_k) != 2k."))
    end
    k = div(two_k, 2)

    half_col_p = copy(@inbounds view(p, :, 1:k))
    mul!((@inbounds view(p, :, 1:k)), -J.λ, @inbounds view(p, :, (k + 1):lastindex(p, 2)))
    mul!((@inbounds view(p, :, (k + 1):lastindex(p, 2))), J.λ, @inbounds view(half_col_p, :, :))
    return p
end

function LinearAlgebra.mul!(A::AbstractVecOrMat, J::SymplecticElement, p::AbstractVecOrMat)
    size_p = size(p)
    two_n = size_p[1]
    if two_n % 2 != 0
        throw(ArgumentError("'p' must have even row dimension, was: $(two_n) != 2n."))
    end
    n = div(two_n, 2)
    k = length(size_p) == 1 ? 0 : div(size_p[2], 2)

    # k == 0 means we're multiplying with a vector:
    @boundscheck k == 0 ? checkbounds(A, 1:(2n), 1) : checkbounds(A, 1:(2n), 1:(2k))

    # Perform left multiply by λ*J:
    mul!((@inbounds view(A, 1:n, :)), J.λ, @inbounds view(p, (n + 1):lastindex(p, 1), :))
    mul!((@inbounds view(A, (n + 1):lastindex(A, 1), :)), -J.λ, @inbounds view(p, 1:n, :))
    return A
end

function LinearAlgebra.mul!(A::AbstractVecOrMat, p::AbstractMatrix, J::SymplecticElement)
    two_n, two_k = size(p)
    if two_k % 2 != 0
        throw(ArgumentError("'p' must have even col dimension, was: $(two_k) != 2k."))
    end
    n = div(two_n, 2)
    k = div(two_k, 2)

    # n == 0 means we're multiplying with a vector:
    @boundscheck n == 0 ? checkbounds(A, 1, 1:(2k)) : checkbounds(A, 1:(2n), 1:(2k))

    # Perform right multiply by λ*J:
    mul!((@inbounds view(A, :, 1:k)), -J.λ, @inbounds view(p, :, (k + 1):lastindex(p, 2)))
    mul!((@inbounds view(A, :, (k + 1):lastindex(A, 2))), J.λ, @inbounds view(p, :, 1:k))
    return A
end

function add_scaled_I!(A::AbstractMatrix, λ::Number)
    LinearAlgebra.checksquare(A)
    @inbounds for i in axes(A, 1)
        A[i, i] += λ
    end
    return A
end
