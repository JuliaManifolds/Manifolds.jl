@doc raw"""
    change_gradient(M::Hyperbolic{n}, ::EuclideanMetric, p::PoincareBallPoint, X::PoincareBallTVector)

Since in the metric we always have the term `` α = \frac{2}{1-\sum_{i=1}^n p_i^2}`` per element,
the correction for the gradient reads `` Z = \frac{1}{α^2}X``.
"""
function change_gradient(
    ::Hyperbolic,
    ::EuclideanMetric,
    p::PoincareBallPoint,
    X::PoincareBallTVector,
)
    α = 2 / (1 - norm(p.value)^2)
    Y = copy(M, p, X)
    Y.value ./= α^2
    return Y
end

@doc raw"""
    change_metric(M::Hyperbolic{n}, ::EuclideanMetric, p::PoincareBallPoint, X::PoincareBallTVector)

Since in the metric we always have the term `` α = \frac{2}{1-\sum_{i=1}^n p_i^2}`` per element,
the correction for the metric reads `` Z = \frac{1}{α}X``.
"""
function change_metric(
    ::Hyperbolic,
    ::EuclideanMetric,
    p::PoincareBallPoint,
    X::PoincareBallTVector,
)
    α = 2 / (1 - norm(p.value)^2)
    Y = copy(M, p, X)
    Y.value ./= α
    return Y
end

function check_point(M::Hyperbolic{N}, p::PoincareBallPoint; kwargs...) where {N}
    mpv = check_point(Euclidean(N), p.value; kwargs...)
    mpv === nothing || return mpv
    if !(norm(p.value) < 1)
        return DomainError(
            norm(p.value),
            "The point $(p) does not lie on $(M) since its norm is not less than 1.",
        )
    end
end

@doc raw"""
    convert(::Type{PoincareBallPoint}, p::HyperboloidPoint)
    convert(::Type{PoincareBallPoint}, p::T) where {T<:AbstractVector}

convert a [`HyperboloidPoint`](@ref) $p∈ℝ^{n+1}$ from the hyperboloid model of the [`Hyperbolic`](@ref)
manifold $\mathcal H^n$ to a [`PoincareBallPoint`](@ref) $π(p)∈ℝ^{n}$ in the Poincaré ball model.
The isometry is defined by

````math
π(p) = \frac{1}{1+p_{n+1}} \begin{pmatrix}p_1\\⋮\\p_n\end{pmatrix}
````

Note that this is also used, when `x` is a vector.
"""
convert(::Type{PoincareBallPoint}, ::Any)
function convert(t::Type{PoincareBallPoint}, p::HyperboloidPoint)
    return convert(t, p.value)
end
function convert(::Type{PoincareBallPoint}, p::T) where {T<:AbstractVector}
    return PoincareBallPoint(1 / (1 + last(p)) .* p[1:(end - 1)])
end

@doc raw"""
    convert(::Type{PoincareBallPoint}, p::PoincareHalfSpacePoint)

convert a point [`PoincareHalfSpacePoint`](@ref) `p` (from $ℝ^n$) from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ to a [`PoincareBallPoint`](@ref) $π(p) ∈ ℝ^n$.
Denote by $\tilde p = (p_1,\ldots,p_{d-1})^{\mathrm{T}}$. Then the isometry is defined by

````math
π(p) = \frac{1}{\lVert \tilde p \rVert^2 + (p_n+1)^2}
\begin{pmatrix}2p_1\\⋮\\2p_{n-1}\\\lVert p\rVert^2 - 1\end{pmatrix}.
````
"""
function convert(::Type{PoincareBallPoint}, p::PoincareHalfSpacePoint)
    return PoincareBallPoint(
        1 / (norm(p.value[1:(end - 1)])^2 + (last(p.value) + 1)^2) .*
        vcat(2 .* p.value[1:(end - 1)], norm(p.value)^2 - 1),
    )
end

@doc raw"""
    convert(::Type{PoincareBallTVector}, p::HyperboloidPoint, X::HyperboloidTVector)
    convert(::Type{PoincareBallTVector}, p::P, X::T) where {P<:AbstractVector, T<:AbstractVector}

convert a [`HyperboloidTVector`](@ref) `X` at `p` to a [`PoincareBallTVector`](@ref)
on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ by computing the push forward $π_*(p)[X]$ of
the isometry $π$ that maps from the Hyperboloid to the Poincaré ball,
cf. [`convert(::Type{PoincareBallPoint}, ::HyperboloidPoint)`](@ref).

The formula reads

````math
π_*(p)[X] = \frac{1}{p_{n+1}+1}\Bigl(\tilde X - \frac{X_{n+1}}{p_{n+1}+1}\tilde p \Bigl),
````

where $\tilde X = \begin{pmatrix}X_1\\⋮\\X_n\end{pmatrix}$
and $\tilde p = \begin{pmatrix}p_1\\⋮\\p_n\end{pmatrix}$.
"""
convert(::Type{PoincareBallTVector}, ::Any)
function convert(t::Type{PoincareBallTVector}, p::HyperboloidPoint, X::HyperboloidTVector)
    return convert(t, convert(AbstractVector, p), convert(AbstractVector, p, X))
end
function convert(
    ::Type{PoincareBallTVector},
    p::P,
    X::T,
) where {P<:AbstractVector,T<:AbstractVector}
    return PoincareBallTVector(
        1 / (p[end] + 1) .* (X[1:(end - 1)] .- (X[end] / (p[end] + 1) .* p[1:(end - 1)])),
    )
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
        (p,X)::Tuple{HyperboloidPoint,HyperboloidTVector}
    )
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
        (p, X)::Tuple{P,T},
    ) where {P<:AbstractVector, T <: AbstractVector}

Convert a [`HyperboloidPoint`](@ref) `p` and a [`HyperboloidTVector`](@ref) `X`
to a [`PoincareBallPoint`](@ref) and a [`PoincareBallTVector`](@ref) simultaneously,
see [`convert(::Type{PoincareBallPoint}, ::HyperboloidPoint)`](@ref) and
[`convert(::Type{PoincareBallTVector}, ::HyperboloidPoint, ::HyperboloidTVector)`](@ref)
for the formulae.
"""
function convert(
    ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
    (p, X)::Tuple{HyperboloidPoint,HyperboloidTVector},
)
    return (convert(PoincareBallPoint, p), convert(PoincareBallTVector, p, X))
end
function convert(
    ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
    (p, X)::Tuple{P,T},
) where {P<:AbstractVector,T<:AbstractVector}
    return (convert(PoincareBallPoint, p), convert(PoincareBallTVector, p, X))
end

@doc raw"""
    convert(
        ::Type{PoincareBallTVector},
        p::PoincareHalfSpacePoint,
        X::PoincareHalfSpaceTVector
    )

convert a [`PoincareHalfSpaceTVector`](@ref) `X` at `p` to a [`PoincareBallTVector`](@ref)
on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ by computing the push forward $π_*(p)[X]$ of
the isometry $π$ that maps from the Poincaré half space to the Poincaré ball,
cf. [`convert(::Type{PoincareBallPoint}, ::PoincareHalfSpacePoint)`](@ref).

The formula reads

````math
π_*(p)[X] =
\frac{1}{\lVert \tilde p\rVert^2 + (1+p_n)^2}
\begin{pmatrix}
2X_1\\
⋮\\
2X_{n-1}\\
2⟨X,p⟩
\end{pmatrix}
-
\frac{2}{(\lVert \tilde p\rVert^2 + (1+p_n)^2)^2}
\begin{pmatrix}
2p_1(⟨X,p⟩+X_n)\\
⋮\\
2p_{n-1}(⟨X,p⟩+X_n)\\
(\lVert p \rVert^2-1)(⟨X,p⟩+X_n)
\end{pmatrix}
````
where $\tilde p = \begin{pmatrix}p_1\\⋮\\p_{n-1}\end{pmatrix}$.
"""
function convert(
    ::Type{PoincareBallTVector},
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTVector,
)
    den = norm(p.value[1:(end - 1)])^2 + (last(p.value) + 1)^2
    scp = dot(p.value, X.value)
    c1 =
        (2 / den .* X.value[1:(end - 1)]) .-
        (4 * (scp + last(X.value)) / (den^2)) .* p.value[1:(end - 1)]
    c2 = 2 * scp / den - 2 * (norm(p.value)^2 - 1) * (scp + last(X.value)) / (den^2)
    return PoincareBallTVector(vcat(c1, c2))
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
        (p,X)::Tuple{HyperboloidPoint,HyperboloidTVector}
    )
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
        (p, X)::Tuple{T,T},
    ) where {T <: AbstractVector}

Convert a [`PoincareHalfSpacePoint`](@ref) `p` and a [`PoincareHalfSpaceTVector`](@ref) `X`
to a [`PoincareBallPoint`](@ref) and a [`PoincareBallTVector`](@ref) simultaneously,
see [`convert(::Type{PoincareBallPoint}, ::PoincareHalfSpacePoint)`](@ref) and
[`convert(::Type{PoincareBallTVector}, ::PoincareHalfSpacePoint, ::PoincareHalfSpaceTVector)`](@ref)
for the formulae.
"""
function convert(
    ::Type{Tuple{PoincareBallPoint,PoincareBallTVector}},
    (p, X)::Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector},
)
    return (convert(PoincareBallPoint, p), convert(PoincareBallTVector, p, X))
end

@doc raw"""
    distance(::Hyperbolic, p::PoincareBallPoint, q::PoincareBallPoint)

Compute the distance on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ represented in the
Poincaré ball model. The formula reads

````math
d_{\mathcal H^n}(p,q) =
\operatorname{acosh}\Bigl(
  1 + \frac{2\lVert p - q \rVert^2}{(1-\lVert p\rVert^2)(1-\lVert q\rVert^2)}
\Bigr)
````
"""
function distance(::Hyperbolic, p::PoincareBallPoint, q::PoincareBallPoint)
    return acosh(
        1 +
        2 * norm(p.value .- q.value)^2 / ((1 - norm(p.value)^2) * (1 - norm(q.value)^2)),
    )
end

@doc raw"""
    inner(::Hyperbolic, p::PoincareBallPoint, X::PoincareBallTVector, Y::PoincareBallTVector)

Compute the inner producz in the Poincaré ball model. The formula reads
````math
g_p(X,Y) = \frac{4}{(1-\lVert p \rVert^2)^2}  ⟨X, Y⟩ .
````
"""
function inner(
    ::Hyperbolic,
    p::PoincareBallPoint,
    X::PoincareBallTVector,
    Y::PoincareBallTVector,
)
    return 4 / (1 - norm(p.value)^2)^2 * dot(X.value, Y.value)
end

@doc raw"""
    project(::Hyperbolic, ::PoincareBallPoint, ::PoincareBallTVector)

projction of tangent vectors in the Poincaré ball model is just the identity, since
the tangent space consists of all $ℝ^n$.
"""
project(::Hyperbolic, ::PoincareBallPoint, ::PoincareBallTVector)

function allocate_result(
    ::Hyperbolic,
    ::typeof(project),
    X::PoincareBallTVector,
    ::PoincareBallPoint,
)
    return PoincareBallTVector(allocate(X.value))
end

function project!(
    ::Hyperbolic,
    Y::PoincareBallTVector,
    ::PoincareBallPoint,
    X::PoincareBallTVector,
)
    return (Y.value .= X.value)
end
