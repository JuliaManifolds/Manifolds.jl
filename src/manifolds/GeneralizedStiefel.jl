@doc raw"""
    GeneralizedStiefel{n,k,𝔽,B} <: AbstractDecoratorManifold{𝔽}

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
several functions like the [`zero_vector`](@ref) are inherited from the embedding.

The manifold is named after
[Eduard L. Stiefel](https://en.wikipedia.org/wiki/Eduard_Stiefel) (1909–1978).

# Constructor
    GeneralizedStiefel(n, k, B=I_n, F=ℝ)

Generate the (real-valued) Generalized Stiefel manifold of $n\times k$ dimensional
orthonormal matrices with scalar product `B`.
"""
struct GeneralizedStiefel{n,k,𝔽,TB<:AbstractMatrix} <: AbstractDecoratorManifold{𝔽}
    B::TB
end

function GeneralizedStiefel(
    n::Int,
    k::Int,
    B::AbstractMatrix=Matrix{Float64}(I, n, n),
    𝔽::AbstractNumbers=ℝ,
)
    return GeneralizedStiefel{n,k,𝔽,typeof(B)}(B)
end

active_traits(f, ::GeneralizedStiefel, args...) = merge_traits(IsEmbeddedManifold())

@doc raw"""
    check_point(M::GeneralizedStiefel, p; kwargs...)

Check whether `p` is a valid point on the [`GeneralizedStiefel`](@ref) `M`=$\operatorname{St}(n,k,B)$,
i.e. that it has the right [`AbstractNumbers`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#number-system) type and $x^{\mathrm{H}}Bx$
is (approximately) the identity, where $\cdot^{\mathrm{H}}$ is the complex conjugate
transpose. The settings for approximately can be set with `kwargs...`.
"""
function check_point(M::GeneralizedStiefel{n,k,𝔽}, p; kwargs...) where {n,k,𝔽}
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
function check_size(M::GeneralizedStiefel{n,k,𝔽}, p) where {n,k,𝔽}
    return check_size(get_embedding(M), p) #avoid embed, since it uses copyto!
end
function check_size(M::GeneralizedStiefel{n,k,𝔽}, p, X) where {n,k,𝔽}
    return check_size(get_embedding(M), p, X) #avoid embed, since it uses copyto!
end

@doc raw"""
    check_vector(M::GeneralizedStiefel, p, X; kwargs...)

Check whether `X` is a valid tangent vector at `p` on the [`GeneralizedStiefel`](@ref)
`M`=$\operatorname{St}(n,k,B)$, i.e. the [`AbstractNumbers`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#number-system) fits,
`p` is a valid point on `M` and
it (approximately) holds that $p^{\mathrm{H}}BX + \overline{X^{\mathrm{H}}Bp} = 0$, where
`kwargs...` is passed to the `isapprox`.
"""
function check_vector(M::GeneralizedStiefel{n,k,𝔽}, p, X; kwargs...) where {n,k,𝔽}
    if !isapprox(p' * M.B * X, -conj(X' * M.B * p); kwargs...)
        return DomainError(
            norm(p' * M.B * X + conj(X' * M.B * p)),
            "The matrix $(X) does not lie in the tangent space of $(p) on $(M), since x'Bv + v'Bx is not the zero matrix.",
        )
    end
    return nothing
end

get_embedding(::GeneralizedStiefel{N,K,𝔽}) where {N,K,𝔽} = Euclidean(N, K; field=𝔽)

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
function manifold_dimension(::GeneralizedStiefel{n,k,ℝ}) where {n,k}
    return n * k - div(k * (k + 1), 2)
end
manifold_dimension(::GeneralizedStiefel{n,k,ℂ}) where {n,k} = 2 * n * k - k * k
manifold_dimension(::GeneralizedStiefel{n,k,ℍ}) where {n,k} = 4 * n * k - k * (2k - 1)

@doc raw"""
    project(M::GeneralizedStiefel,p)

Project `p` from the embedding onto the [`GeneralizedStiefel`](@ref) `M`, i.e. compute `q`
as the polar decomposition of $p$ such that $q^{\mathrm{H}}Bq$ is the identity,
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project(::GeneralizedStiefel, ::Any)

function project!(M::GeneralizedStiefel, q, p)
    s = svd(p)
    e = eigen(s.U' * M.B * s.U)
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

where $\operatorname{Sym}(y)$ is the symmetrization of $y$, e.g. by
$\operatorname{Sym}(y) = \frac{y^{\mathrm{H}}+y}{2}$.
"""
project(::GeneralizedStiefel, ::Any, ::Any)

function project!(M::GeneralizedStiefel, Y, p, X)
    A = p' * M.B' * X
    copyto!(Y, X)
    mul!(Y, p, Hermitian((A .+ A') ./ 2), -1, 1)
    return Y
end
@doc raw"""
    retract(M::GeneralizedStiefel, p, X)
    retract(M::GeneralizedStiefel, p, X, ::PolarRetraction)
    retract(M::GeneralizedStiefel, p, X, ::ProjectionRetraction)

Compute the SVD-based retraction [`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction) on the
[`GeneralizedStiefel`](@ref) manifold `M`, which in this case is the same as
the projection based retraction employing the exponential map in the embedding
and projecting the result back to the manifold.

The default retraction for this manifold is the [`ProjectionRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.ProjectionRetraction).
"""
retract(::GeneralizedStiefel, ::Any...)

default_retraction_method(::GeneralizedStiefel) = ProjectionRetraction()

function retract_polar!(M::GeneralizedStiefel, q, p, X, t::Number)
    q .= p .+ t .* X
    project!(M, q, q)
    return q
end
function retract_project!(M::GeneralizedStiefel, q, p, X, t::Number)
    q .= p .+ t .* X
    project!(M, q, q)
    return q
end

function Base.show(io::IO, M::GeneralizedStiefel{n,k,𝔽}) where {n,k,𝔽}
    return print(io, "GeneralizedStiefel($(n), $(k), $(M.B), $(𝔽))")
end
