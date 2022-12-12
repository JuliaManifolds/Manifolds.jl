@doc raw"""
    Symplectic{n, 𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}

The symplectic manifold consists of all ``2n \times 2n`` matrices which preserve
the canonical symplectic form over ``𝔽^{2n × 2n} \times 𝔽^{2n × 2n}``,
````math
    \omega\colon 𝔽^{2n × 2n} \times 𝔽^{2n × 2n} \rightarrow 𝔽,
    \quad \omega(x, y) = p^{\mathrm{T}} Q_{2n} q, \; x, y \in 𝔽^{2n × 2n},
````
where
````math
Q_{2n} =
\begin{bmatrix}
  0_n & I_n \\
 -I_n & 0_n
\end{bmatrix}.
````
That is, the symplectic manifold consists of
````math
\operatorname{Sp}(2n, ℝ) = \bigl\{ p ∈ ℝ^{2n × 2n} \, \big| \, p^{\mathrm{T}}Q_{2n}p = Q_{2n} \bigr\},
````
with ``0_n`` and ``I_n`` denoting the ``n × n`` zero-matrix
and indentity matrix in ``ℝ^{n \times n}`` respectively.

The tangent space at a point ``p`` is given by [^BendokatZimmermann2021]
````math
\begin{align*}
    T_p\operatorname{Sp}(2n)
        &= \{X \in \mathbb{R}^{2n \times 2n} \;|\; p^{T}Q_{2n}X + X^{T}Q_{2n}p = 0 \}, \\
        &= \{X = pQS \;|\; S ∈ R^{2n × 2n}, S^{\mathrm{T}} = S \}.
\end{align*}
````

# Constructor
    Symplectic(2n, field=ℝ) -> Symplectic{div(2n, 2), field}()

Generate the (real-valued) symplectic manifold of ``2n \times 2n`` symplectic matrices.
The constructor for the [`Symplectic`](@ref) manifold accepts the even column/row embedding
dimension ``2n`` for the real symplectic manifold, ``ℝ^{2n × 2n}``.
"""
struct Symplectic{n,𝔽} <: AbstractDecoratorManifold{𝔽} end

function active_traits(f, ::Symplectic, args...)
    return merge_traits(IsEmbeddedManifold(), IsDefaultMetric(RealSymplecticMetric()))
end

function Symplectic(n::Int, field::AbstractNumbers=ℝ)
    n % 2 == 0 || throw(ArgumentError("The dimension of the symplectic manifold
                        embedding space must be even. Was odd, n % 2 == $(n % 2)."))
    return Symplectic{div(n, 2),field}()
end

@doc raw"""
    RealSymplecticMetric <: RiemannianMetric

The canonical Riemannian metric on the symplectic manifold,
defined pointwise for ``p \in \operatorname{Sp}(2n)`` by [^Fiori2011]
````math
\begin{align*}
    & g_p \colon T_p\operatorname{Sp}(2n) \times T_p\operatorname{Sp}(2n) \rightarrow ℝ, \\
    & g_p(Z_1, Z_2) = \operatorname{tr}((p^{-1}Z_1)^{\mathrm{T}} (p^{-1}Z_2)).
\end{align*}
````
This metric is also the default metric for the [`Symplectic`](@ref) manifold.
"""
struct RealSymplecticMetric <: RiemannianMetric end

@doc raw"""
    ExtendedSymplecticMetric <: AbstractMetric

The extension of the [`RealSymplecticMetric`](@ref) at a point `p \in \operatorname{Sp}(2n)`
as an inner product over the embedding space ``ℝ^{2n \times 2n}``, i.e.
````math
    \langle x, y \rangle_{p} = \langle p^{-1}x, p^{-1}\rangle_{\operatorname{Fr}}
    = \operatorname{tr}(x^{\mathrm{T}}(pp^{\mathrm{T}})^{-1}y), \;\forall\; x, y \in ℝ^{2n \times 2n}.
````
"""
struct ExtendedSymplecticMetric <: AbstractMetric end

@doc raw"""
    SymplecticMatrix{T}

A lightweight structure to represent the action of the matrix
representation of the canonical symplectic form,
````math
Q_{2n}(λ) = λ
\begin{bmatrix}
0_n & I_n \\
 -I_n & 0_n
\end{bmatrix} \quad \in ℝ^{2n \times 2n},
````
such that the canonical symplectic form is represented by
````math
\omega_{2n}(x, y) = x^{\mathrm{T}}Q_{2n}(1)y, \quad x, y \in ℝ^{2n}.
````

The entire matrix is however not instantiated in memory, instead a scalar
``λ`` of type `T` is stored, which is used to keep track of scaling and transpose operations
applied  to each `SymplecticMatrix`.
For example, given `Q = SymplecticMatrix(1.0)` represented as `1.0*[0 I; -I 0]`,
the adjoint `Q'` returns `SymplecticMatrix(-1.0) = (-1.0)*[0 I; -I 0]`.
"""
struct SymplecticMatrix{T}
    λ::T
end
SymplecticMatrix() = SymplecticMatrix(1)
SymplecticMatrix(λ::T) where {T<:Number} = SymplecticMatrix{T}(λ)

function SymplecticMatrix(arrays::Vararg{AbstractArray})
    TS = Base.promote_type(map(eltype, arrays)...)
    return SymplecticMatrix(one(TS))
end

@doc raw"""
    change_representer(::Symplectic, ::EuclideanMetric, p, X)
    change_representer!(::Symplectic, Y, ::EuclideanMetric, p, X)

Compute the representation of a tangent vector ``ξ ∈ T_p\operatorname{Sp}(2n, ℝ)`` s.t.
````math
    g_p(c_p(ξ), η) = ⟨ξ, η⟩^{\text{Euc}} \;∀\; η ∈ T_p\operatorname{Sp}(2n, ℝ).
````
with the conversion function
````math
    c_p : T_p\operatorname{Sp}(2n, ℝ) \rightarrow T_p\operatorname{Sp}(2n, ℝ), \quad
    c_p(ξ) = \frac{1}{2} pp^{\mathrm{T}} ξ + \frac{1}{2} pQ ξ^{\mathrm{T}} pQ.
````

Each of the terms ``c_p^1(ξ) = p p^{\mathrm{T}} ξ`` and ``c_p^2(ξ) = pQ ξ^{\mathrm{T}} pQ`` from the
above definition of ``c_p(η)`` are themselves metric compatible in the sense that
````math
    c_p^i : T_p\operatorname{Sp}(2n, ℝ) \rightarrow \mathbb{R}^{2n \times 2n}\quad
    g_p^i(c_p(ξ), η) = ⟨ξ, η⟩^{\text{Euc}} \;∀\; η ∈ T_p\operatorname{Sp}(2n, ℝ),
````
for ``i \in {1, 2}``. However the range of each function alone is not confined to
``T_p\operatorname{Sp}(2n, ℝ)``, but the convex combination
````math
    c_p(ξ) = \frac{1}{2}c_p^1(ξ) + \frac{1}{2}c_p^2(ξ)
````
does have the correct range ``T_p\operatorname{Sp}(2n, ℝ)``.
"""
change_representer(::Symplectic, ::EuclideanMetric, p, X)

function change_representer!(::Symplectic, Y, ::EuclideanMetric, p, X)
    Q = SymplecticMatrix(p, X)
    pT_X = p' * X
    Y .= (1 / 2) .* p * (pT_X .+ Q * pT_X' * Q)
    return Y
end

@doc raw"""
    change_representer(MetMan::MetricManifold{𝔽, Euclidean{Tuple{m, n}, 𝔽}, ExtendedSymplecticMetric},
                       EucMet::EuclideanMetric, p, X)
    change_representer!(MetMan::MetricManifold{𝔽, Euclidean{Tuple{m, n}, 𝔽}, ExtendedSymplecticMetric},
                        Y, EucMet::EuclideanMetric, p, X)

Change the representation of a matrix ``ξ ∈ \mathbb{R}^{2n \times 2n}``
into the inner product space ``(ℝ^{2n \times 2n}, g_p)`` where the inner product
is given by
``g_p(ξ, η) = \langle p^{-1}ξ, p^{-1}η \rangle = \operatorname{tr}(ξ^{\mathrm{T}}(pp^{\mathrm{T}})^{-1}η)``,
as the extension of the [`RealSymplecticMetric`](@ref) onto the entire embedding space.

By changing the representation we mean to apply a mapping
````math
    c_p : \mathbb{R}^{2n \times 2n} \rightarrow \mathbb{R}^{2n \times 2n},
````
defined by requiring that it satisfy the metric compatibility condition
````math
    g_p(c_p(ξ), η) = ⟨p^{-1}c_p(ξ), p^{-1}η⟩ = ⟨ξ, η⟩^{\text{Euc}}
        \;∀\; η ∈ T_p\operatorname{Sp}(2n, ℝ).
````
In this case, we compute the mapping
````math
    c_p(ξ) = pp^{\mathrm{T}} ξ.
````
"""
function change_representer(
    ::MetricManifold{𝔽,Euclidean{Tuple{m,n},𝔽},ExtendedSymplecticMetric},
    ::EuclideanMetric,
    p,
    X,
) where {𝔽,m,n}
    return p * p' * X
end

function change_representer!(
    ::MetricManifold{𝔽,Euclidean{Tuple{m,n},𝔽},ExtendedSymplecticMetric},
    Y,
    ::EuclideanMetric,
    p,
    X,
) where {𝔽,m,n}
    Y .= p * p' * X
    return Y
end

@doc raw"""
    check_point(M::Symplectic, p; kwargs...)

Check whether `p` is a valid point on the [`Symplectic`](@ref) `M`=$\operatorname{Sp}(2n)$,
i.e. that it has the right [`AbstractNumbers`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#number-system) type and $p^{+}p$ is (approximately)
the identity, where $A^{+} = Q_{2n}^{\mathrm{T}}A^{\mathrm{T}}Q_{2n}$ is the symplectic inverse, with
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

@doc raw"""
    check_vector(M::Symplectic, p, X; kwargs...)

Checks whether `X` is a valid tangent vector at `p` on the [`Symplectic`](@ref)
`M`=``\operatorname{Sp}(2n)``, i.e. the [`AbstractNumbers`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#number-system) fits and
it (approximately) holds that ``p^{T}Q_{2n}X + X^{T}Q_{2n}p = 0``,
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

ManifoldsBase.default_inverse_retraction_method(::Symplectic) = CayleyInverseRetraction()

ManifoldsBase.default_retraction_method(::Symplectic) = CayleyRetraction()

@doc raw"""
    distance(M::Symplectic, p, q)

Compute an approximate geodesic distance between two Symplectic matrices
``p, q \in \operatorname{Sp}(2n)``, as done in [^WangSunFiori2018].
````math
    \operatorname{dist}(p, q)
        ≈ ||\operatorname{Log}(p^+q)||_{\operatorname{Fr}},
````
where the ``\operatorname{Log}(\cdot)`` operator is the matrix logarithm.

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
    X \in T_p\operatorname{Sp},
````
and with the geodesic distance between ``p`` and ``q`` given by
``\operatorname{dist}(p, q) = ||X||_p = ||p^+X||_{\operatorname{Fr}}``
we see that
````math
    \begin{align*}
    ||\operatorname{Log}(p^+q)||_{\operatorname{Fr}}
    &= ||\operatorname{Log}\left(
        \operatorname{Exp}((p^{+}X)^{\mathrm{T}})
        \operatorname{Exp}(p^{+}X - (p^{+}X)^{\mathrm{T}})
    \right)||_{\operatorname{Fr}} \\
    &= ||p^{+}X + \frac{1}{2}[(p^{+}X)^{\mathrm{T}}, p^{+}X - (p^{+}X)^{\mathrm{T}}]
            + \ldots ||_{\operatorname{Fr}} \\
    &≈ ||p^{+}X||_{\operatorname{Fr}} = \operatorname{dist}(p, q).
    \end{align*}
````
"""
function distance(M::Symplectic{n}, p, q) where {n}
    return norm(log(symplectic_inverse_times(M, p, q)))
end

embed(::Symplectic, p) = p
embed(::Symplectic, p, X) = X

@doc raw"""
    exp(M::Symplectic, p, X)
    exp!(M::Symplectic, q, p, X)

The Exponential mapping on the Symplectic manifold with the
[`RealSymplecticMetric`](@ref) Riemannian metric.

For the point ``p \in \operatorname{Sp}(2n)`` the exponential mapping along the tangent
vector ``X \in T_p\operatorname{Sp}(2n)`` is computed as [^WangSunFiori2018]
````math
    \operatorname{exp}_p(X) = p \operatorname{Exp}((p^{-1}X)^{\mathrm{T}})
                                \operatorname{Exp}(p^{-1}X - (p^{-1}X)^{\mathrm{T}}),
````
where ``\operatorname{Exp}(\cdot)`` denotes the matrix exponential.

[^WangSunFiori2018]:
    > Wang, Jing and Sun, Huafei and Fiori, Simone:
    > A Riemannian-steepest-descent approach for optimization on the real symplectic group,
    > Mathematical Methods in the Applied Sciences, 41(11) pp. 4273-4286, 2018
    > doi [10.1002/mma.4890](https://doi.org/10.1002/mma.4890)
"""
exp(::Symplectic, ::Any...)

function exp!(M::Symplectic, q, p, X)
    p_star_X = symplectic_inverse_times(M, p, X)
    q .= p * exp(Array(p_star_X')) * exp(p_star_X - p_star_X')
    return q
end

get_embedding(::Symplectic{n,ℝ}) where {n} = Euclidean(2n, 2n; field=ℝ)

@doc raw"""
    gradient(M::Symplectic, f, p, backend::RiemannianProjectionBackend;
             extended_metric=true)
    gradient!(M::Symplectic, f, p, backend::RiemannianProjectionBackend;
             extended_metric=true)

Compute the manifold gradient ``\text{grad}f(p)`` of a scalar function
``f \colon \operatorname{Sp}(2n) \rightarrow ℝ`` at
``p \in \operatorname{Sp}(2n)``.

The element ``\text{grad}f(p)`` is found as the Riesz representer of the differential
``\text{D}f(p) \colon T_p\operatorname{Sp}(2n) \rightarrow ℝ`` w.r.t.
the Riemannian metric inner product at ``p`` [^Fiori2011].
That is, ``\text{grad}f(p) \in T_p\operatorname{Sp}(2n)`` solves the relation
````math
    g_p(\text{grad}f(p), X) = \text{D}f(p) \quad\forall\; X \in T_p\operatorname{Sp}(2n).
````

The default behaviour is to first change the representation of the Euclidean gradient from
the Euclidean metric to the [`RealSymplecticMetric`](@ref) at ``p``, and then we projecting
the result onto the correct tangent tangent space ``T_p\operatorname{Sp}(2n, ℝ)``
w.r.t the Riemannian metric ``g_p`` extended to the entire embedding space.

# Arguments:
- `extended_metric = true`: If `true`, compute the gradient ``\text{grad}f(p)`` by
    first changing the representer of the Euclidean gradient of a smooth extension
    of ``f``, ``∇f(p)``, w.r.t. the [`RealSymplecticMetric`](@ref) at ``p``
    extended to the entire embedding space, before projecting onto the correct
    tangent vector space w.r.t. the same extended metric ``g_p``.
    If `false`, compute the gradient by first projecting ``∇f(p)`` onto the
    tangent vector space, before changing the representer in the tangent
    vector space to comply with the [`RealSymplecticMetric`](@ref).


[^Fiori2011]:
    > Simone Fiori:
    > Solving minimal-distance problems over the manifold of real-symplectic matrices,
    > SIAM Journal on Matrix Analysis and Applications 32(3), pp. 938-968, 2011.
    > doi [10.1137/100817115](https://doi.org/10.1137/100817115).
"""
function gradient(
    M::Symplectic,
    f,
    p,
    backend::RiemannianProjectionBackend;
    extended_metric=true,
)
    Y = allocate_result(M, gradient, p)
    return gradient!(M, f, Y, p, backend; extended_metric=extended_metric)
end

function gradient!(
    M::Symplectic,
    f,
    X,
    p,
    backend::RiemannianProjectionBackend;
    extended_metric=true,
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
    inner(::Symplectic{n, ℝ}, p, X, Y)

Compute the canonical Riemannian inner product [`RealSymplecticMetric`](@ref)
````math
    g_p(X, Y) = \operatorname{tr}((p^{-1}X)^{\mathrm{T}} (p^{-1}Y))
````
between the two tangent vectors ``X, Y \in T_p\operatorname{Sp}(2n)``.
"""
function inner(M::Symplectic{n,ℝ}, p, X, Y) where {n}
    p_star = inv(M, p)
    return dot((p_star * X), (p_star * Y))
end

@doc raw"""
    inv(::Symplectic, A)
    inv!(::Symplectic, A)

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
A^{+} := Q_{2n}^{\mathrm{T}} A^{\mathrm{T}} Q_{2n},
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
  A_{2, 2}^{\mathrm{T}} & -A_{1, 2}^{\mathrm{T}} \\[1.2mm]
 -A_{2, 1}^{\mathrm{T}} &  A_{1, 1}^{\mathrm{T}}
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
    inverse_retract(M::Symplectic, p, q, ::CayleyInverseRetraction)

Compute the Cayley Inverse Retraction ``X = \mathcal{L}_p^{\operatorname{Sp}}(q)``
such that the Cayley Retraction from ``p`` along ``X`` lands at ``q``, i.e.
``\mathcal{R}_p(X) = q`` [^BendokatZimmermann2021].

First, recall the definition the standard symplectic matrix
````math
Q =
\begin{bmatrix}
 0    & I \\
-I  & 0
\end{bmatrix}
````
as well as the symplectic inverse of a matrix ``A``, ``A^{+} = Q^{\mathrm{T}} A^{\mathrm{T}} Q``.

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

[^BendokatZimmermann2021]:
    > Bendokat, Thomas and Zimmermann, Ralf:
	> The real symplectic Stiefel and Grassmann manifolds: metrics, geodesics and applications
	> arXiv preprint arXiv:[2108.12447](https://arxiv.org/abs/2108.12447), 2021.
"""
inverse_retract(::Symplectic, p, q, ::CayleyInverseRetraction)

function inverse_retract_caley!(M::Symplectic, X, p, q)
    U_inv = lu(add_scaled_I!(symplectic_inverse_times(M, p, q), 1))
    V_inv = lu(add_scaled_I!(symplectic_inverse_times(M, q, p), 1))

    X .= 2 .* ((p / V_inv .- p / U_inv) .+ ((p + q) / U_inv) .- p)
    return X
end

@doc raw"""
    manifold_dimension(::Symplectic{n})

Returns the dimension of the symplectic manifold
embedded in ``ℝ^{2n \times 2n}``, i.e.
````math
    \operatorname{dim}(\operatorname{Sp}(2n)) = (2n + 1)n.
````
"""
manifold_dimension(::Symplectic{n}) where {n} = (2n + 1) * n

@doc raw"""
    project(::Symplectic, p, A)
    project!(::Symplectic, Y, p, A)

Given a point ``p \in \operatorname{Sp}(2n)``,
project an element ``A \in \mathbb{R}^{2n \times 2n}`` onto
the tangent space ``T_p\operatorname{Sp}(2n)`` relative to
the euclidean metric of the embedding ``\mathbb{R}^{2n \times 2n}``.

That is, we find the element ``X \in T_p\operatorname{SpSt}(2n, 2k)``
which solves the constrained optimization problem
````math
    \operatorname{min}_{X \in \mathbb{R}^{2n \times 2n}} \frac{1}{2}||X - A||^2, \quad
    \text{s.t.}\;
    h(X) \colon= X^{\mathrm{T}} Q p + p^{\mathrm{T}} Q X = 0,
````
where ``h\colon\mathbb{R}^{2n \times 2n} \rightarrow \operatorname{skew}(2n)`` defines
the restriction of ``X`` onto the tangent space ``T_p\operatorname{SpSt}(2n, 2k)``.
"""
project(::Symplectic, p, A)

function project!(::Symplectic, Y, p, A)
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
    project!(::MetricManifold{𝔽,Euclidean,ExtendedSymplecticMetric}, Y, p, X) where {𝔽}

Compute the projection of ``X ∈ R^{2n × 2n}`` onto ``T_p\operatorname{Sp}(2n, ℝ)`` w.r.t.
the Riemannian metric ``g`` [`RealSymplecticMetric`](@ref).
The closed form projection mapping is given by [^GaoSonAbsilStykel2021]
````math
    \operatorname{P}^{T_p\operatorname{Sp}(2n)}_{g_p}(X) = pQ\operatorname{sym}(p^{\mathrm{T}}Q^{\mathrm{T}}X),
````
where ``\operatorname{sym}(A) = \frac{1}{2}(A + A^{\mathrm{T}})``.
This function is not exported.
"""
function project!(
    ::MetricManifold{𝔽,Euclidean{Tuple{m,n},𝔽},ExtendedSymplecticMetric},
    Y,
    p,
    X,
) where {m,n,𝔽}
    Q = SymplecticMatrix(p, X)

    pT_QT_X = p' * Q' * X
    symmetrized_pT_QT_X = (1 / 2) .* (pT_QT_X + pT_QT_X')

    Y .= p * Q * (symmetrized_pT_QT_X)
    return Y
end

@doc raw"""
    project_normal!(::MetricManifold{𝔽,Euclidean,ExtendedSymplecticMetric}, Y, p, X)

Project onto the normal of the tangent space ``(T_p\operatorname{Sp}(2n))^{\perp_g}`` at
a point ``p ∈ \operatorname{Sp}(2n)``, relative to the riemannian metric
``g`` [`RealSymplecticMetric`](@ref).
That is,
````math
(T_p\operatorname{Sp}(2n))^{\perp_g} = \{Y \in \mathbb{R}^{2n \times 2n} :
                        g_p(Y, X) = 0 \;\forall\; X \in T_p\operatorname{Sp}(2n)\}.
````
The closed form projection operator onto the normal space is given by [^GaoSonAbsilStykel2021]
````math
\operatorname{P}^{(T_p\operatorname{Sp}(2n))\perp}_{g_p}(X) = pQ\operatorname{skew}(p^{\mathrm{T}}Q^{\mathrm{T}}X),
````
where ``\operatorname{skew}(A) = \frac{1}{2}(A - A^{\mathrm{T}})``.
This function is not exported.

[^GaoSonAbsilStykel2021]:
    > Gao, Bin and Son, Nguyen Thanh and Absil, P-A and Stykel, Tatjana:
    > Riemannian optimization on the symplectic Stiefel manifold,
    > SIAM Journal on Optimization 31(2), pp. 1546-1575, 2021.
    > doi [10.1137/20M1348522](https://doi.org/10.1137/20M1348522)
"""
function project_normal!(
    ::MetricManifold{𝔽,Euclidean{Tuple{m,n},𝔽},ExtendedSymplecticMetric},
    Y,
    p,
    X,
) where {m,n,𝔽}
    Q = SymplecticMatrix(p, X)

    pT_QT_X = p' * Q' * X
    skew_pT_QT_X = (1 / 2) .* (pT_QT_X .- pT_QT_X')

    Y .= p * Q * skew_pT_QT_X
    return Y
end

@doc raw"""
    rand(::SymplecticStiefel; vector_at=nothing,
        hamiltonian_norm = (vector_at === nothing ? 1/2 : 1.0))

Generate a random point on ``\operatorname{Sp}(2n)`` or a random
tangent vector ``X \in T_p\operatorname{Sp}(2n)`` if `vector_at` is set to
a point ``p \in \operatorname{Sp}(2n)``.

A random point on ``\operatorname{Sp}(2n)`` is constructed by generating a
random Hamiltonian matrix ``Ω \in \mathfrak{sp}(2n,F)`` with norm `hamiltonian_norm`,
and then transforming it to a symplectic matrix by applying the Cayley transform
````math
    \operatorname{cay}\colon \mathfrak{sp}(2n,F) \rightarrow \operatorname{Sp}(2n),
    \; \Omega \mapsto (I - \Omega)^{-1}(I + \Omega).
````
To generate a random tangent vector in ``T_p\operatorname{Sp}(2n)``, this code employs the
second tangent vector space parametrization of [Symplectic](@ref).
It first generates a random symmetric matrix ``S`` by `S = randn(2n, 2n)`
and then symmetrizes it as `S = S + S'`.
Then ``S`` is normalized to have Frobenius norm of `hamiltonian_norm`
and `X = pQS` is returned, where `Q` is the [`SymplecticMatrix`](@ref).
"""
function Base.rand(
    M::Symplectic;
    vector_at=nothing,
    hamiltonian_norm=(vector_at === nothing ? 1 / 2 : 1.0),
)
    if vector_at === nothing
        Ω = rand_hamiltonian(M; frobenius_norm=hamiltonian_norm)
        return (I - Ω) \ (I + Ω)
    else
        random_vector(M, vector_at; symmetric_norm=hamiltonian_norm)
    end
end

function random_vector(::Symplectic{n}, p::AbstractMatrix; symmetric_norm=1.0) where {n}
    # Generate random symmetric matrix:
    S = randn(2n, 2n)
    S .= (S + S')
    S *= symmetric_norm / norm(S)
    Q = SymplecticMatrix(p)
    lmul!(Q, S)
    return p * S
end

function rand_hamiltonian(::Symplectic{n}; frobenius_norm=1.0) where {n}
    A = randn(n, n)
    B = randn(n, n)
    C = randn(n, n)
    B = (1 / 2) .* (B .+ B')
    C = (1 / 2) .* (C .+ C')
    Ω = [A B; C -A']
    return frobenius_norm * Ω / norm(Ω, 2)
end

@doc raw"""
    retract(::Symplectic, p, X, ::CayleyRetraction)
    retract!(::Symplectic, q, p, X, ::CayleyRetraction)

Compute the Cayley retraction on ``p ∈ \operatorname{Sp}(2n, ℝ)`` in
the direction of tangent vector ``X ∈ T_p\operatorname{Sp}(2n, ℝ)``,
as defined in by Birtea et al in proposition 2 [^BirteaCaşuComănescu2020].

Using the symplectic inverse of a matrix ``A \in ℝ^{2n \times 2n}``,
``
A^{+} := Q_{2n}^{\mathrm{T}} A^{\mathrm{T}} Q_{2n}
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

[^BirteaCaşuComănescu2020]:
    > Birtea, Petre and Caşu, Ioan and Comănescu, Dan:
    > Optimization on the real symplectic group,
    > Monatshefte f{\"u}r Mathematik, Springer, 2020.
    > doi [10.1007/s00605-020-01369-9](https://doi.org/10.1007/s00605-020-01369-9)
"""
retract(M::Symplectic, p, X)

function retract_caley!(M::Symplectic, q, p, X)
    p_star_X = symplectic_inverse_times(M, p, X)

    divisor = lu(2 * I - p_star_X)
    q .= p * (divisor \ add_scaled_I!(p_star_X, 2.0))
    return q
end

Base.show(io::IO, ::Symplectic{n,𝔽}) where {n,𝔽} = print(io, "Symplectic{$(2n), $(𝔽)}()")

@doc raw"""
    symplectic_inverse_times(::Symplectic, p, q)
    symplectic_inverse_times!(::Symplectic, A, p, q)

Directly compute the symplectic inverse of ``p \in \operatorname{Sp}(2n)``,
multiplied with ``q \in \operatorname{Sp}(2n)``.
That is, this function efficiently computes
``p^+q = (Q_{2n}p^{\mathrm{T}}Q_{2n})q \in ℝ^{2n \times 2n}``,
where ``Q_{2n}`` is the [`SymplecticMatrix`](@ref)
of size ``2n \times 2n``.
"""
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

ndims(Q::SymplecticMatrix) = 2
copy(Q::SymplecticMatrix) = SymplecticMatrix(copy(Q.λ))
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

function (Base.:^)(Q::SymplecticMatrix, n::Integer)
    return ifelse(
        n % 2 == 0,
        UniformScaling((-1)^(div(n, 2)) * (Q.λ)^n),
        SymplecticMatrix((-1)^(div(n - 1, 2)) * (Q.λ)^n),
    )
end

(Base.:*)(x::Number, Q::SymplecticMatrix) = SymplecticMatrix(x * Q.λ)
(Base.:*)(Q::SymplecticMatrix, x::Number) = SymplecticMatrix(x * Q.λ)
function (Base.:*)(Q1::SymplecticMatrix, Q2::SymplecticMatrix)
    return LinearAlgebra.UniformScaling(-Q1.λ * Q2.λ)
end

Base.transpose(Q::SymplecticMatrix) = -Q
Base.adjoint(Q::SymplecticMatrix) = SymplecticMatrix(-conj(Q.λ))
Base.inv(Q::SymplecticMatrix) = SymplecticMatrix(-(1 / Q.λ))

(Base.:+)(Q1::SymplecticMatrix, Q2::SymplecticMatrix) = SymplecticMatrix(Q1.λ + Q2.λ)
(Base.:-)(Q1::SymplecticMatrix, Q2::SymplecticMatrix) = SymplecticMatrix(Q1.λ - Q2.λ)

(Base.:+)(Q::SymplecticMatrix, p::AbstractMatrix) = p + Q
function (Base.:+)(p::AbstractMatrix, Q::SymplecticMatrix)
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
    TS = typeof(one(eltype(p)) + one(eltype(Q)))
    out = copyto!(similar(p, TS), p)

    add_scaled_I!(view(out, 1:n, (n + 1):(2n)), Q.λ)
    add_scaled_I!(view(out, (n + 1):(2n), 1:n), -Q.λ)
    return out
end

# Binary minus:
(Base.:-)(Q::SymplecticMatrix, p::AbstractMatrix) = Q + (-p)
(Base.:-)(p::AbstractMatrix, Q::SymplecticMatrix) = p + (-Q)

function (Base.:*)(Q::SymplecticMatrix, p::AbstractVecOrMat)
    two_n = size(p)[1]
    if two_n % 2 != 0
        throw(ArgumentError("'p' must have even row dimension, was: $(two_n) != 2n."))
    end
    n = div(two_n, 2)

    # Allocate new memory:
    TS = typeof(one(eltype(p)) + one(eltype(Q)))
    Qp = similar(p, TS)

    # Perform left mulitply by λ*Q:
    mul!((@inbounds view(Qp, 1:n, :)), Q.λ, @inbounds view(p, (n + 1):lastindex(p, 1), :))
    mul!((@inbounds view(Qp, (n + 1):lastindex(Qp, 1), :)), -Q.λ, @inbounds view(p, 1:n, :))

    return Qp
end

function (Base.:*)(p::AbstractMatrix, Q::SymplecticMatrix)
    two_k = size(p)[2]
    if two_k % 2 != 0
        throw(ArgumentError("'p' must have even column dimension, was: $(two_k) != 2k."))
    end
    k = div(two_k, 2)

    # Allocate new memory:
    TS = typeof(one(eltype(p)) + one(eltype(Q)))
    pQ = similar(p, TS)

    # Perform right mulitply by λ*Q:
    mul!((@inbounds view(pQ, :, 1:k)), -Q.λ, @inbounds view(p, :, (k + 1):lastindex(p, 2)))
    mul!((@inbounds view(pQ, :, (k + 1):lastindex(pQ, 2))), Q.λ, @inbounds view(p, :, 1:k))
    return pQ
end

function LinearAlgebra.lmul!(Q::SymplecticMatrix, p::AbstractVecOrMat)
    # Perform left multiplication by a symplectic matrix,
    # overwriting the matrix p in place:
    two_n = size(p)[1]
    if two_n % 2 != 0
        throw(ArgumentError("'p' must have even row dimension, was: $(two_n) != 2n."))
    end
    n = div(two_n, 2)

    half_row_p = copy(@inbounds view(p, 1:n, :))

    mul!((@inbounds view(p, 1:n, :)), Q.λ, @inbounds view(p, (n + 1):lastindex(p, 1), :))

    mul!(
        (@inbounds view(p, (n + 1):lastindex(p, 1), :)),
        -Q.λ,
        @inbounds view(half_row_p, :, :)
    )
    return p
end

function LinearAlgebra.rmul!(p::AbstractMatrix, Q::SymplecticMatrix)
    # Perform right multiplication by a symplectic matrix,
    # overwriting the matrix p in place:
    two_k = size(p)[2]
    if two_k % 2 != 0
        throw(ArgumentError("'p' must have even column dimension, was: $(two_k) != 2k."))
    end
    k = div(two_k, 2)

    half_col_p = copy(@inbounds view(p, :, 1:k))

    mul!((@inbounds view(p, :, 1:k)), -Q.λ, @inbounds view(p, :, (k + 1):lastindex(p, 2)))

    mul!(
        (@inbounds view(p, :, (k + 1):lastindex(p, 2))),
        Q.λ,
        @inbounds view(half_col_p, :, :)
    )

    return p
end

function LinearAlgebra.mul!(A::AbstractVecOrMat, Q::SymplecticMatrix, p::AbstractVecOrMat)
    size_p = size(p)
    two_n = size_p[1]
    if two_n % 2 != 0
        throw(ArgumentError("'p' must have even row dimension, was: $(two_n) != 2n."))
    end
    n = div(two_n, 2)
    k = length(size_p) == 1 ? 0 : div(size_p[2], 2)

    # k == 0 means we're multiplying with a vector:
    @boundscheck k == 0 ? checkbounds(A, 1:(2n), 1) : checkbounds(A, 1:(2n), 1:(2k))

    # Perform left multiply by λ*Q:
    mul!((@inbounds view(A, 1:n, :)), Q.λ, @inbounds view(p, (n + 1):lastindex(p, 1), :))
    mul!((@inbounds view(A, (n + 1):lastindex(A, 1), :)), -Q.λ, @inbounds view(p, 1:n, :))
    return A
end

function LinearAlgebra.mul!(A::AbstractVecOrMat, p::AbstractMatrix, Q::SymplecticMatrix)
    two_n, two_k = size(p)
    if two_k % 2 != 0
        throw(ArgumentError("'p' must have even col dimension, was: $(two_k) != 2k."))
    end
    n = div(two_n, 2)
    k = div(two_k, 2)

    # n == 0 means we're multiplying with a vector:
    @boundscheck n == 0 ? checkbounds(A, 1, 1:(2k)) : checkbounds(A, 1:(2n), 1:(2k))

    # Perform right multiply by λ*Q:
    mul!((@inbounds view(A, :, 1:k)), -Q.λ, @inbounds view(p, :, (k + 1):lastindex(p, 2)))
    mul!((@inbounds view(A, :, (k + 1):lastindex(A, 2))), Q.λ, @inbounds view(p, :, 1:k))
    return A
end

function add_scaled_I!(A::AbstractMatrix, λ::Number)
    LinearAlgebra.checksquare(A)
    @inbounds for i in axes(A, 1)
        A[i, i] += λ
    end
    return A
end
