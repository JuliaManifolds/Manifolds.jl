@doc doc"""
    GeneralizedStiefel{n,k,B,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultEmbeddingType}

The Generalized Stiefel manifold consists of all $n\times k$, $n\geq k$ orthonormal
matrices w.r.t. an arbitrary scalar product with symmetric positive definite matrix
$B\in R^{n × n}$, i.e.

````math
\operatorname{St}(n,k,B) = \bigl\{ p \in \mathbb F^{n × k}\ \big|\ p^{\mathrm{H}} B p = I_k \bigr\},
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
    GeneralizedStiefel(n, k, B=I_n, F=ℝ)

Generate the (real-valued) Generalized Stiefel manifold of $n\times k$ dimensional
orthonormal matrices with scalar product `B`.
"""
struct GeneralizedStiefel{n,k,TB<:AbstractMatrix,𝔽} <:
       AbstractEmbeddedManifold{𝔽,DefaultEmbeddingType}
    B::TB
end

function GeneralizedStiefel(
    n::Int,
    k::Int,
    B::AbstractMatrix = Matrix{Float64}(I, n, n),
    F::AbstractNumbers = ℝ,
)
    return GeneralizedStiefel{n,k,typeof(B),F}(B)
end

@doc doc"""
    check_manifold_point(M::GeneralizedStiefel, p; kwargs...)

Check whether `p` is a valid point on the [`GeneralizedStiefel`](@ref) `M`=$\operatorname{St}(n,k,B)$,
i.e. that it has the right [`AbstractNumbers`](@ref) type and $x^{\mathrm{H}}Bx$
is (approximately) the identity, where $\cdot^{\mathrm{H}}$ is the complex conjugate
transpose. The settings for approximately can be set with `kwargs...`.
"""
function check_manifold_point(M::GeneralizedStiefel{n,k,B,𝔽}, p; kwargs...) where {n,k,B,𝔽}
    mpv = invoke(check_manifold_point, Tuple{supertype(typeof(M)), typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    c = p' * M.B * p
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(p) does not lie on $(M), because x'Bx is not the unit matrix.",
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
function check_tangent_vector(
    M::GeneralizedStiefel{n,k,B,𝔽},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {n,k,B,𝔽}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)), typeof(p), typeof(X)},
        M,
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...
    )
    mpv === nothing || return mpv
    if !isapprox(p' * M.B * X + X' * M.B * p, zeros(k, k); kwargs...)
        return DomainError(
            norm(p' * M.B * X + X' * M.B * p),
            "The matrix $(X) does not lie in the tangent space of $(p) on $(M), since x'Bv + v'Bx is not the zero matrix.",
        )
    end
end

decorated_manifold(M::GeneralizedStiefel{N,K,B,𝔽}) where {N,K,B,𝔽} = Euclidean(N, K; field = 𝔽)

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
function manifold_dimension(::GeneralizedStiefel{n,k,B,ℝ}) where {n,k,B}
    return n * k - div(k * (k + 1), 2)
end
manifold_dimension(::GeneralizedStiefel{n,k,B,ℂ}) where {n,k,B} = 2 * n * k - k * k
manifold_dimension(::GeneralizedStiefel{n,k,B,ℍ}) where {n,k,B} = 4 * n * k - k * (2k - 1)

@doc doc"""
    project(M::GeneralizedStiefel,p)

Project `p` from the embedding onto the [`GeneralizedStiefel`](@ref) `M`, i.e. compute `q`
as the polar decomposition of $p$ such that $q^{\mathrm{H}}Bq$ is the identity,
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project(::GeneralizedStiefel, ::Any)

function project!(M::GeneralizedStiefel, q, p)
    s = svd(p)
    e = eigen(s.U' * M.B * s.U)
    qsinv = e.vectors * Diagonal(1 ./ sqrt.(e.values))
    q .= s.U * qsinv * e.vectors' * s.V'
    return q
end

@doc doc"""
    project(M:GeneralizedStiefel, p, X)

Project `X` onto the tangent space of `p` to the [`GeneralizedStiefel`](@ref) manifold `M`.
The formula reads

````math
\operatorname{proj}_{\operatorname{St}(n,k)}(p,X) = X - p\operatorname{Sym}(p^{\mathrm{H}}BX),
````

where $\operatorname{Sym}(y)$ is the symmetrization of $y$, e.g. by
$\operatorname{Sym}(y) = \frac{y^{\mathrm{H}}+y}{2}$.
"""
project(::GeneralizedStiefel, ::Any, ::Any)

function project!(M::GeneralizedStiefel, Y, p, X)
    A = p' * M.B' * X
    copyto!(Y, X - p * Hermitian((A + A') / 2))
    return Y
end
@doc doc"""
    retract(M::GeneralizedStiefel, p, X)
    retract(M::GeneralizedStiefel, p, X, ::PolarRetraction)
    retract(M::GeneralizedStiefel, p, X, ::ProjectionRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`GeneralizedStiefel`](@ref) manifold `M`, which in this case is the same as
the projection based retraction employing the exponential map in the embedding
and projecting the result back to the manifold.

The default retraction for this manifold is the [`ProjectionRetraction`](@ref).
"""
retract(::GeneralizedStiefel, ::Any...)
retract(M::GeneralizedStiefel, p, X) = retract(M, p, X, ProjectionRetraction())

retract!(M::GeneralizedStiefel, q, p, X) = retract!(M, q, p, X, ProjectionRetraction())
function retract!(M::GeneralizedStiefel, q, p, X, ::PolarRetraction)
    project!(M, q, p + X)
    return q
end
function retract!(M::GeneralizedStiefel, q, p, X, ::ProjectionRetraction)
    project!(M, q, p + X)
    return q
end

function show(io::IO, M::GeneralizedStiefel{n,k,B,F}) where {n,k,B,F}
    print(io, "GeneralizedStiefel($(n), $(k), $(M.B), $(F))")
end


@doc doc"""
    vector_transport_to(M::GeneralizedStiefel, p, X, q, ::ProjectionTransport)

Compute the vector transport of the tangent vector `X` at `p` to `q`,
using the [`project`](@ref project(::GeneralizedStiefel, ::Any...))
of `X` to `q`.
"""
vector_transport_to(::GeneralizedStiefel, ::Any, ::Any, ::Any, ::ProjectionTransport)

function vector_transport_to!(M::GeneralizedStiefel, Y, p, X, q, ::ProjectionTransport)
    project!(M, Y, q, X)
    return Y
end
