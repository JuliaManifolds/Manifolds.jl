@doc doc"""
    GeneralizedStiefel{n,k,T} <: AbstractEmbeddedManifold{AbstractIsometricEmbeddingType}

The Generalized Stiefel manifold consists of all $n\times k$, $n\geq k$ orthonormal matrices w.r.t. an arbitrary scalar product `B`, i.e.

````math
\operatorname{St}(n,k,B) = \{ p \in \mathbb F^{n × k} : p^{\mathrm{H}} B p = I_k \},
````

where $𝔽 ∈ \{ℝ, ℂ\}$,
$\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian, and
$I_k \in \mathbb R^{k × k}$ denotes the $k × k$ identity matrix.


In the case $B=I_k$ one gets the usual [`Stiefel`](@ref) manifold.

The tangent space at a point $p\in\mathcal M=\operatorname{St}(n,k,B)$ is given by

````math
T_p\mathcal M = \{ X \in 𝔽^{n × k} : p^{\mathrm{H}}BX + X^{\mathrm{H}}Bp=0_n\},
````
where $0_k$ is the $k × k$ zero matrix.

This manifold is modeled as an embedded manifold to the [`Euclidean`](@ref), i.e.
several functions like the [`zero_tangent_vector`](@ref) are inherited from the embedding.

The manifold is named after
[Eduard L. Stiefel](https://en.wikipedia.org/wiki/Eduard_Stiefel) (1909–1978).

# Constructor
    GeneralizedStiefel(n,k,B=I_k,F=ℝ)

Generate the (real-valued) Generalized Stiefel manifold of $n\times k$ dimensional
orthonormal matrices with scalar product `B`.
"""
struct GeneralizedStiefel{n,k,F,TB<:AbstractMatrix} <: AbstractEmbeddedManifold{AbstractIsometricEmbeddingType}
    B::TB
end

GeneralizedStiefel(n::Int, k::Int, B::AbstractMatrix = Matrix{Float64}(I,n,n), F::AbstractNumbers = ℝ) = GeneralizedStiefel{n,k,F,typeof(B)}(B)

base_manifold(M::GeneralizedStiefel) = M
decorated_manifold(M::GeneralizedStiefel{N,K}) where {N,K} = Euclidean(N,K; field=ℝ)

@doc doc"""
    check_manifold_point(M::GeneralizedStiefel, p; kwargs...)

Check whether `p` is a valid point on the [`GeneralizedStiefel`](@ref) `M`=$\operatorname{St}(n,k,B)$,
i.e. that it has the right [`AbstractNumbers`](@ref) type and $x^{\mathrm{H}}Bx$
is (approximately) the identity, where $\cdot^{\mathrm{H}}$ is the complex conjugate
transpose. The settings for approximately can be set with `kwargs...`.
"""
function check_manifold_point(M::GeneralizedStiefel{n,k,T}, p; kwargs...) where {n,k,T}
    if (T === ℝ) && !(eltype(p) <: Real)
        return DomainError(
            eltype(p),
            "The matrix $(p) is not a real-valued matrix, so it does not lie on the Generalized Stiefel manifold of dimension ($(n),$(k)).",
        )
    end
    if (T === ℂ) && !(eltype(p) <: Real) && !(eltype(x) <: Complex)
        return DomainError(
            eltype(p),
            "The matrix $(p) is neiter real- nor complex-valued matrix, so it does not lie on the complex Generalized Stiefel manifold of dimension ($(n),$(k)).",
        )
    end
    if any(size(p) != representation_size(M))
        return DomainError(
            size(p),
            "The matrix $(p) is does not lie on the Generalized Stiefel manifold of dimension ($(n),$(k)), since its dimensions are wrong.",
        )
    end
    c = p' * M.B * p
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(p) does not lie on the Generalized Stiefel manifold of dimension ($(n),$(k)), because x'Bx is not the unit matrix.",
        )
    end
end


@doc doc"""
    check_tangent_vector(M::GeneralizedStiefel, p, X; kwargs...)

Check whether `X` is a valid tangent vector at `p` on the [`GeneralizedStiefel`](@ref)
`M`=$\operatorname{St}(n,k,B)$, i.e. the [`AbstractNumbers`](@ref) fits,
`p` is a valid point on `M` and
it (approximately) holds that $p^{\mathrm{H}}BX + X^{\mathrm{H}}Bp = 0$, where
`kwargs...` is passed to the `isapprox`.
"""
function check_tangent_vector(M::GeneralizedStiefel{n,k,T}, p, X; kwargs...) where {n,k,T}
    mpe = check_manifold_point(M, p, kwargs)
    mpe === nothing || return mpe
    if (T === ℝ) && !(eltype(X) <: Real)
        return DomainError(
            eltype(X),
            "The matrix $(X) is not a real-valued matrix, so it can not be a tangent vector to the Generalized Stiefel manifold of dimension ($(n),$(k)).",
        )
    end
    if (T === ℂ) && !(eltype(X) <: Real) && !(eltype(v) <: Complex)
        return DomainError(
            eltype(X),
            "The matrix $(X) is a neither real- nor complex-valued matrix, so it can not be a tangent vector to the complex Generalized Stiefel manifold of dimension ($(n),$(k)).",
        )
    end
    if any(size(X) != representation_size(M))
        return DomainError(
            size(X),
            "The matrix $(X) does not lie in the tangent space of $(p) on the Generalized Stiefel manifold of dimension ($(n),$(k)), since its dimensions are wrong.",
        )
    end
    if !isapprox(p' * M.B * X + X' * M.B * p, zeros(k, k); kwargs...)
        return DomainError(
            norm(p' * M.B * X + X' * M.B * p),
            "The matrix $(X) does not lie in the tangent space of $(p) on the Generalized Stiefel manifold of dimension ($(n),$(k)), since x'Bv + v'Bx is not the zero matrix.",
        )
    end
end

embed!(::GeneralizedStiefel, q, p) = (q .= p)

embed!(::GeneralizedStiefel, Y, p, X) = (Y .= X)

@doc doc"""
    inner(M::GeneralizedStiefel, p, X, Y)

Compute the inner product for two tangent vectors `X`, `Y` from the tangent space of `p`
on the [`GeneralizedStiefel`](@ref) manifold `M`. The formula reads

````math
(X, Y)_p = \operatorname{trace}(v^{\mathrm{H}}Bw),
````
i.e. the metric induced by the scalar product `B` from the embedding, restricted to the
tangent space.
"""
inner(M::GeneralizedStiefel, p, X, Y) = dot(X, M.B * Y)


@doc doc"""
    manifold_dimension(M::GeneralizedStiefel)

Return the dimension of the [`GeneralizedStiefel`](@ref) manifold `M`=$\operatorname{St}(n,k,B,𝔽)$.
The dimension is given by

````math
\begin{aligned}
\dim \mathrm{St}(n, k, B, ℝ) &= nk - \frac{1}{2}k(k+1) \\
\dim \mathrm{St}(n, k, B, ℂ) &= 2nk - k^2\\
\dim \mathrm{St}(n, k, B, ℍ) &= 4nk - k(2k-1)
\end{aligned}
````
"""
manifold_dimension(::GeneralizedStiefel{n,k,ℝ}) where {n,k} = n * k - div(k * (k + 1), 2)
manifold_dimension(::GeneralizedStiefel{n,k,ℂ}) where {n,k} = 2 * n * k - k * k
manifold_dimension(::GeneralizedStiefel{n,k,ℍ}) where {n,k} = 4 * n * k - k * (2k - 1)

@doc doc"""
    project_point(M::GeneralizedStiefel,p)

Projects `p` from the embedding onto the [`GeneralizedStiefel`](@ref) `M`, i.e. compute `q`
as the polar decomposition of $p$ such that $q^{\mathrm{H}}Bq$ is the identity,
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project_point(::GeneralizedStiefel, ::Any...)

function project_point!(M::GeneralizedStiefel, q, p)
    s = svd(p)
    e=eigen(s.U' * M.B * s.U)
    qsinv = e.vectors * Diagonal(1 ./ sqrt.(e.values))
    q .= s.U * qsinv * e.vectors' * s.V'
    return q
end

@doc doc"""
    project_tangent(M, p, X)

Project `X` onto the tangent space of `p` to the [`GeneralizedStiefel`](@ref) manifold `M`.
The formula reads

````math
\operatorname{proj}_{\operatorname{St}(n,k)}(p,X) = X - p\operatorname{Sym}(p^{\mathrm{H}}BX),
````

where $\operatorname{Sym}(y)$ is the symmetrization of $y$, e.g. by
$\operatorname{Sym}(y) = \frac{y^{\mathrm{H}}+y}{2}$.
"""
project_tangent(::GeneralizedStiefel, ::Any...)

project_tangent!(::GeneralizedStiefel, Y, p, X) = copyto!(Y, X - p * Symmetric(B*p'*X))

@doc doc"""
    retract(M, p, X, ::PolarRetraction)
    retract(M, p, X, ::ProjectionRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`GeneralizedStiefel`](@ref) manifold `M`, which in this case is the same as
the projection based retraction employing the exponential map in the embedding
and projecting the result back to the manifold.
"""
retract(::GeneralizedStiefel, ::Any...)

function retract!(M::GeneralizedStiefel, q, p, X, ::PolarRetraction)
    project_point!(M, q, p+X)
    return q
end
function retract!(M::GeneralizedStiefel, y, x, v, ::ProjectionRetraction)
    project_point!(M, q, p+X)
    return y
end

@doc doc"""
    vector_transport_to(M, p, X, q, ::ProjectionTransport)

Compute the vector transport of the tangent vector `X` at `p` to `q`,
using the [`project_tangent`](@ref project_tangent(::GeneralizedStiefel, ::Any...))
of `X` to `q`.
"""
vector_transport_to(::GeneralizedStiefel, ::Any, ::Any, ::Any, ::Any, ::ProjectionTransport)

function vector_transport_to!(M::GeneralizedStiefel, Y, p, X, q, ::ProjectionTransport)
    project_tangent!(Y,q,X)
    return Y
end