@doc raw"""
    GeneralizedStiefel{𝔽,T,B} <: AbstractDecoratorManifold{𝔽}

The Generalized Stiefel manifold consists of all ``n×k``, ``n\geq k`` orthonormal
matrices w.r.t. an arbitrary scalar product with symmetric positive definite matrix
``B\in R^{n×n}``, i.e.

````math
\operatorname{St}(n,k,B) = \bigl\{ p \in \mathbb F^{n×k}\ \big|\ p^{\mathrm{H}} B p = I_k \bigr\},
````

where ``𝔽 ∈ \{ℝ, ℂ\}``,
``⋅^{\mathrm{H}}`` denotes the complex conjugate transpose or Hermitian, and
``I_k \in \mathbb R^{k×k}`` denotes the ``k×k`` identity matrix.


In the case ``B=I_k`` one gets the usual [`Stiefel`](@ref) manifold.

The tangent space at a point ``p\in\mathcal M=\operatorname{St}(n,k,B)`` is given by

````math
T_p\mathcal M = \{ X \in 𝔽^{n×k} : p^{\mathrm{H}}BX + X^{\mathrm{H}}Bp=0_n\},
````
where ``0_k`` is the ``k×k`` zero matrix.

This manifold is modeled as an embedded manifold to the [`Euclidean`](@ref), i.e.
several functions like the [`zero_vector`](@ref) are inherited from the embedding.

The manifold is named after
[Eduard L. Stiefel](https://en.wikipedia.org/wiki/Eduard_Stiefel) (1909–1978).

# Constructor
    GeneralizedStiefel(n, k, B=I_n, F=ℝ; parameter::Symbol=:type)

Generate the (real-valued) Generalized Stiefel manifold of ``n×k`` dimensional
orthonormal matrices with scalar product `B`.

`parameter`: whether a type parameter should be used to store `n` and `k`. By default size
is stored in type. Value can either be `:field` or `:type`.
"""
struct GeneralizedStiefel{𝔽, T, TB <: AbstractMatrix} <: AbstractDecoratorManifold{𝔽}
    size::T
    B::TB
end

function GeneralizedStiefel(
        n::Int,
        k::Int,
        B::AbstractMatrix = Matrix{Float64}(I, n, n),
        𝔽::AbstractNumbers = ℝ;
        parameter::Symbol = :type,
    )
    size = wrap_type_parameter(parameter, (n, k))
    return GeneralizedStiefel{𝔽, typeof(size), typeof(B)}(size, B)
end

@doc raw"""
    check_point(M::GeneralizedStiefel, p; kwargs...)

Check whether `p` is a valid point on the [`GeneralizedStiefel`](@ref) `M`=``\operatorname{St}(n,k,B)``,
i.e. that it has the right [`AbstractNumbers`](@extref ManifoldsBase number-system) type and ``x^{\mathrm{H}}Bx``
is (approximately) the identity, where ``⋅^{\mathrm{H}}`` is the complex conjugate
transpose. The settings for approximately can be set with `kwargs...`.
"""
function check_point(M::GeneralizedStiefel, p; kwargs...)
    c = p' * M.B * p
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(p) does not lie on $(M), because x'Bx is not the unit matrix.",
        )
    end
    return nothing
end

# overwrite passing to embedding
function check_size(M::GeneralizedStiefel, p::P) where {P}
    return check_size(get_embedding(M, P), p) #avoid embed, since it uses copyto!
end
function check_size(M::GeneralizedStiefel, p::P, X) where {P}
    return check_size(get_embedding(M, P), p, X) #avoid embed, since it uses copyto!
end

@doc raw"""
    check_vector(M::GeneralizedStiefel, p, X; atol, rtol, kwargs...)

Check whether `X` is a valid tangent vector at `p` on the [`GeneralizedStiefel`](@ref)
`M`=``\operatorname{St}(n,k,B)``.
This requires that the [`AbstractNumbers`](@extref ManifoldsBase number-system) fits,
`p` is a valid point on `M` and it (approximately) holds that
``p^{\mathrm{H}}BX + X^{\mathrm{H}}Bp = 0``, that is, the norm of the left hand side is at most
`atol` or at most `rtol` times ``2\lVert X \rVert_p``, the largest value this norm can attain.
"""
function check_vector(
        M::GeneralizedStiefel, p, X::T;
        atol::Real = sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
        rtol::Real = sqrt(eps(real(float(number_eltype(T))))),
        kwargs...,
    ) where {T}
    r = norm(p' * M.B * X + X' * M.B * p)
    if !(r <= atol || r <= rtol * 2 * norm(M, p, X))
        return DomainError(
            r,
            "The matrix $(X) does not lie in the tangent space of $(p) on $(M), since x'Bv + v'Bx is not the zero matrix.",
        )
    end
    return nothing
end

function get_embedding(::GeneralizedStiefel{𝔽, TypeParameter{Tuple{n, k}}}) where {n, k, 𝔽}
    return Euclidean(n, k; field = 𝔽)
end
function get_embedding(M::GeneralizedStiefel{𝔽, Tuple{Int, Int}}) where {𝔽}
    n, k = get_parameter(M.size)
    return Euclidean(n, k; field = 𝔽, parameter = :field)
end

function ManifoldsBase.get_embedding_type(::GeneralizedStiefel)
    return ManifoldsBase.EmbeddedManifoldType()
end

@doc raw"""
    inner(M::GeneralizedStiefel, p, X, Y)

Compute the inner product for two tangent vectors `X`, `Y` from the tangent space of `p`
on the [`GeneralizedStiefel`](@ref) manifold `M`. The formula reads

````math
(X, Y)_p = \operatorname{trace}(v^{\mathrm{H}}Bw),
````
i.e. the metric induced by the scalar product `B` from the embedding, restricted to the
tangent space.
"""
inner(M::GeneralizedStiefel, p, X, Y) = dot(X, M.B, Y)

"""
    is_flat(M::GeneralizedStiefel)

Return true if [`GeneralizedStiefel`](@ref) `M` is one-dimensional.
"""
is_flat(M::GeneralizedStiefel) = manifold_dimension(M) == 1

@doc raw"""
    manifold_dimension(M::GeneralizedStiefel)

Return the dimension of the [`GeneralizedStiefel`](@ref) manifold `M`=``\operatorname{St}(n,k,B,𝔽)``.
The dimension is given by

````math
\begin{aligned}
\dim \mathrm{St}(n, k, B, ℝ) &= nk - \frac{1}{2}k(k+1) \\
\dim \mathrm{St}(n, k, B, ℂ) &= 2nk - k^2\\
\dim \mathrm{St}(n, k, B, ℍ) &= 4nk - k(2k-1)
\end{aligned}
````
"""
function manifold_dimension(M::GeneralizedStiefel{ℝ})
    n, k = get_parameter(M.size)
    return n * k - div(k * (k + 1), 2)
end
function manifold_dimension(M::GeneralizedStiefel{ℂ})
    n, k = get_parameter(M.size)
    return 2 * n * k - k * k
end
function manifold_dimension(M::GeneralizedStiefel{ℍ})
    n, k = get_parameter(M.size)
    return 4 * n * k - k * (2k - 1)
end

@doc raw"""
    project(M::GeneralizedStiefel, p)

Project `p` from the embedding onto the [`GeneralizedStiefel`](@ref) `M`, i.e. compute `q`
as the polar decomposition of ``p`` such that ``q^{\mathrm{H}}Bq`` is the identity,
where ``⋅^{\mathrm{H}}`` denotes the hermitian, i.e. complex conjugate transposed.
"""
project(::GeneralizedStiefel, ::Any)

function project!(M::GeneralizedStiefel, q, p)
    s = svd(p)
    e = eigen(Hermitian(s.U' * M.B * s.U))
    qsinv = e.vectors ./ sqrt.(transpose(e.values))
    q .= s.U * qsinv * e.vectors' * s.V'
    return q
end

@doc raw"""
    project(M:GeneralizedStiefel, p, X)

Project `X` onto the tangent space of `p` to the [`GeneralizedStiefel`](@ref) manifold `M`.
The formula reads

````math
\operatorname{proj}_{\operatorname{St}(n,k)}(p,X) = X - p\operatorname{Sym}(p^{\mathrm{H}}BX),
````

where ``\operatorname{Sym}(y)`` is the symmetrization of ``y``, e.g. by
``\operatorname{Sym}(y) = \frac{y^{\mathrm{H}}+y}{2}``.
"""
project(::GeneralizedStiefel, ::Any, ::Any)

function project!(M::GeneralizedStiefel, Y, p, X)
    A = p' * M.B' * X
    copyto!(Y, X)
    mul!(Y, p, Hermitian((A .+ A') ./ 2), -1, 1)
    return Y
end

@doc raw"""
    rand(::GeneralizedStiefel; vector_at=nothing, σ::Real=1.0)

When `vector_at` is `nothing`, return a random (Gaussian) point `p` on the [`GeneralizedStiefel`](@ref) manifold `M`.
This generates a (Gaussian) matrix of size ``n×k`` with standard deviation `σ` and returns its
(generalized) orthogonalized version, i.e. the projection onto the manifold of the
Q component of its QR decomposition.

When `vector_at` is not `nothing`, return a (Gaussian) random vector from the tangent space
``T_{vector\_at}\mathrm{St}(n,k)`` with mean zero and standard deviation `σ` by projecting a
random Matrix onto the tangent vector at `vector_at`.
"""
rand(::GeneralizedStiefel; σ::Real = 1.0)

function Random.rand!(
        rng::AbstractRNG,
        M::GeneralizedStiefel,
        pX;
        vector_at = nothing,
        σ::Real = one(real(eltype(pX))),
    )
    n, k = get_parameter(M.size)
    if vector_at === nothing
        A = σ * randn(rng, eltype(pX), n, k)
        project!(M, pX, Matrix(qr(A).Q))
    else
        Z = σ * randn(rng, eltype(pX), size(pX))
        project!(M, pX, vector_at, Z)
        normalize!(pX)
    end
    return pX
end

@doc raw"""
    retract(M::GeneralizedStiefel, p, X, ::ProjectionRetraction)

Compute the projection based retraction on the [`GeneralizedStiefel`](@ref) manifold `M`,
which employs the exponential map in the embedding and projects the result back to the manifold,

````math
\operatorname{retr}_p X = U(U^{\mathrm{H}}BU)^{-1/2}V^{\mathrm{H}},
````

where ``UΣV^{\mathrm{H}} = p + X`` is the singular value decomposition.
"""
retract(::GeneralizedStiefel, ::Any, ::Any, ::ProjectionRetraction)

"""
    default_retraction_method(M::GeneralizedStiefel)

Return [`ProjectionRetraction`](@extref `ManifoldsBase.ProjectionRetraction`) as the default retraction for the
[`GeneralizedStiefel`](@ref) manifold.
"""
default_retraction_method(::GeneralizedStiefel) = ProjectionRetraction()

@doc raw"""
    retract(M::GeneralizedStiefel, p, X, ::PolarRetraction)

Compute the [`PolarRetraction`](@extref `ManifoldsBase.PolarRetraction`) on the [`GeneralizedStiefel`](@ref)
manifold `M`. For real matrices this is the polar decomposition of ``p + X`` with respect to ``B``,

````math
\operatorname{retr}_p X = (p + X)(I_k + X^{\mathrm{T}}BX)^{-1/2},
````

see Eq. (3.3) in [ShustinAvron:2023](@cite).
For complex matrices it is the projection of ``p + X`` onto the manifold.
"""
retract(::GeneralizedStiefel, ::Any, ::Any, ::PolarRetraction)

function ManifoldsBase.retract_polar!(M::GeneralizedStiefel, q, p, X)
    return ManifoldsBase.retract_polar_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_polar_fused!(M::GeneralizedStiefel, q, p, X, t::Number)
    q .= p .+ t .* X
    project!(M, q, q)
    return q
end
function ManifoldsBase.retract_polar_fused!(M::GeneralizedStiefel{ℝ}, q, p, X, t::Number)
    q .= (p .+ t .* X) / sqrt(Symmetric(I + t^2 * (X' * M.B * X)))
    return q
end

@doc raw"""
    inverse_retract(M::GeneralizedStiefel{ℝ}, p, q, ::PolarInverseRetraction)

Compute the inverse of the [`PolarRetraction`](@extref `ManifoldsBase.PolarRetraction`) on the real
[`GeneralizedStiefel`](@ref) manifold `M` for `q` close enough to `p`,

````math
\operatorname{retr}_p^{-1} q = qZ - p,
````

where ``Z`` is the symmetric positive definite solution of the Lyapunov equation
``p^{\mathrm{T}}BqZ + Zq^{\mathrm{T}}Bp = 2I_k``, see Eqs. (3.4) and (3.5) in [ShustinAvron:2023](@cite).
"""
inverse_retract(::GeneralizedStiefel{ℝ}, ::Any, ::Any, ::PolarInverseRetraction)

function inverse_retract_polar!(M::GeneralizedStiefel{ℝ}, X, p, q)
    Z = lyap(p' * M.B * q, -2 * one(p' * p))
    mul!(X, q, Z)
    X .-= p
    return X
end

@doc raw"""
    retract(M::GeneralizedStiefel{ℝ}, p, X, ::QRRetraction)

Compute the [`QRRetraction`](@extref `ManifoldsBase.QRRetraction`) on the real [`GeneralizedStiefel`](@ref)
manifold `M`, the QR decomposition of ``p + X`` with respect to ``B``,

````math
\operatorname{retr}_p X = (p + X)R^{-1},
````

where ``R^{\mathrm{T}}R = (p + X)^{\mathrm{T}}B(p + X)`` is the Cholesky decomposition,
see Eq. (3.6) in [ShustinAvron:2023](@cite).
"""
retract(::GeneralizedStiefel{ℝ}, ::Any, ::Any, ::QRRetraction)

function ManifoldsBase.retract_qr!(M::GeneralizedStiefel{ℝ}, q, p, X)
    return ManifoldsBase.retract_qr_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_qr_fused!(M::GeneralizedStiefel{ℝ}, q, p, X, t::Number)
    q .= p .+ t .* X
    R = cholesky(Symmetric(q' * M.B * q)).U
    return rdiv!(q, R)
end

@doc raw"""
    inverse_retract(M::GeneralizedStiefel{ℝ}, p, q, ::QRInverseRetraction)

Compute the inverse of the [`QRRetraction`](@extref `ManifoldsBase.QRRetraction`) on the real
[`GeneralizedStiefel`](@ref) manifold `M` for `q` close enough to `p`,

````math
\operatorname{retr}_p^{-1} q = qR - p,
````

where ``R`` is the upper triangular solution with positive diagonal of
``p^{\mathrm{T}}BqR + R^{\mathrm{T}}q^{\mathrm{T}}Bp = 2I_k``, see Eqs. (3.7) and (3.8) in [ShustinAvron:2023](@cite).
"""
inverse_retract(::GeneralizedStiefel{ℝ}, ::Any, ::Any, ::QRInverseRetraction)

function inverse_retract_qr!(M::GeneralizedStiefel{ℝ}, X, p, q)
    n, k = get_parameter(M.size)
    _stiefel_inv_retr_qr_mul_by_r!(Stiefel(n, k), X, q, p' * M.B * q, eltype(X))
    X .-= p
    return X
end

@doc raw"""
    retract(M::GeneralizedStiefel{ℝ}, p, X, ::CayleyRetraction)

Compute the [`CayleyRetraction`](@extref `ManifoldsBase.CayleyRetraction`) on the real [`GeneralizedStiefel`](@ref)
manifold `M`, the Cayley transform with respect to ``B``,

````math
\operatorname{retr}_p X = \Bigl(I_n - \frac{1}{2}W\Bigr)^{-1}\Bigl(I_n + \frac{1}{2}W\Bigr)p,
\qquad
W = \Bigl(I_n - \frac{1}{2}pp^{\mathrm{T}}B\Bigr)Xp^{\mathrm{T}}B - pX^{\mathrm{T}}\Bigl(I_n - \frac{1}{2}Bpp^{\mathrm{T}}\Bigr)B,
````

see Eq. (3.9) in [ShustinAvron:2023](@cite).
"""
retract(::GeneralizedStiefel{ℝ}, ::Any, ::Any, ::CayleyRetraction)

function ManifoldsBase.retract_pade!(M::GeneralizedStiefel{ℝ}, q, p, X, m::PadeRetraction{1})
    return ManifoldsBase.retract_pade_fused!(M, q, p, X, one(eltype(p)), m)
end
function ManifoldsBase.retract_pade_fused!(
        M::GeneralizedStiefel{ℝ}, q, p, X, t::Number, ::PadeRetraction{1},
    )
    tX = t * X
    W = (I - p * p' * M.B / 2) * tX * p' * M.B - p * tX' * (I - M.B * p * p' / 2) * M.B
    return copyto!(q, (I - W / 2) \ ((I + W / 2) * p))
end

function ManifoldsBase.retract_project!(M::GeneralizedStiefel, q, p, X)
    return ManifoldsBase.retract_project_fused!(M, q, p, X, one(eltype(p)))
end
function ManifoldsBase.retract_project_fused!(M::GeneralizedStiefel, q, p, X, t::Number)
    q .= p .+ t .* X
    project!(M, q, q)
    return q
end

function Base.show(io::IO, M::GeneralizedStiefel{𝔽, TypeParameter{Tuple{n, k}}}) where {n, k, 𝔽}
    return print(io, "GeneralizedStiefel($(n), $(k), $(M.B), $(𝔽))")
end
function Base.show(io::IO, M::GeneralizedStiefel{𝔽, Tuple{Int, Int}}) where {𝔽}
    n, k = get_parameter(M.size)
    return print(io, "GeneralizedStiefel($(n), $(k), $(M.B), $(𝔽); parameter=:field)")
end
