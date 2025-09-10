@doc raw"""
    change_representer(M::Hyperbolic, ::EuclideanMetric, p::PoincareBallPoint, X::PoincareBallTangentVector)

Since in the metric we have the term `` α = \frac{2}{1-\sum_{i=1}^n p_i^2}`` per element,
the correction for the gradient reads `` Y = \frac{1}{α^2}X``.
"""
change_representer(
    ::Hyperbolic,
    ::EuclideanMetric,
    ::PoincareBallPoint,
    ::PoincareBallTangentVector,
)

function change_representer!(
        ::Hyperbolic,
        Y::PoincareBallTangentVector,
        ::EuclideanMetric,
        p::PoincareBallPoint,
        X::PoincareBallTangentVector,
    )
    α = 2 / (1 - norm(p.value)^2)
    Y.value .= X.value ./ α^2
    return Y
end

@doc raw"""
    change_metric(M::Hyperbolic, ::EuclideanMetric, p::PoincareBallPoint, X::PoincareBallTangentVector)

Since in the metric we always have the term `` α = \frac{2}{1-\sum_{i=1}^n p_i^2}`` per element,
the correction for the metric reads ``Z = \frac{1}{α}X``.
"""
change_metric(
    ::Hyperbolic,
    ::EuclideanMetric,
    ::PoincareBallPoint,
    ::PoincareBallTangentVector,
)

function change_metric!(
        ::Hyperbolic,
        Y::PoincareBallTangentVector,
        ::EuclideanMetric,
        p::PoincareBallPoint,
        X::PoincareBallTangentVector,
    )
    α = 2 / (1 - norm(p.value)^2)
    Y.value .= X.value ./ α
    return Y
end

function check_point(M::Hyperbolic, p::PoincareBallPoint; kwargs...)
    if !(norm(p.value) < 1)
        return DomainError(
            norm(p.value),
            "The point $(p) does not lie on $(M) since its norm is not less than 1.",
        )
    end
end

function check_size(M::Hyperbolic, p::PoincareBallPoint)
    N = get_parameter(M.size)[1]
    if size(p.value, 1) != N
        !(norm(p.value) < 1)
        return DomainError(
            size(p.value, 1),
            "The point $p does not lie on $M since its length is not $N.",
        )
    end
end

function check_size(
        M::Hyperbolic,
        p::PoincareBallPoint,
        X::PoincareBallTangentVector;
        kwargs...,
    )
    N = get_parameter(M.size)[1]
    if size(X.value, 1) != N
        return DomainError(
            size(X.value, 1),
            "The tangent vector $X can not be a tangent vector for $M since its length is not $N.",
        )
    end
end

@doc raw"""
    convert(::Type{PoincareBallPoint}, p::HyperboloidPoint)
    convert(::Type{PoincareBallPoint}, p::T) where {T<:AbstractVector}

convert a [`HyperboloidPoint`](@ref) ``p∈ℝ^{n+1}`` from the hyperboloid model of the [`Hyperbolic`](@ref)
manifold ``\mathcal H^n`` to a [`PoincareBallPoint`](@ref) ``π(p)∈ℝ^{n}`` in the Poincaré ball model.
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
function convert(::Type{PoincareBallPoint}, p::T) where {T <: AbstractVector}
    return PoincareBallPoint(1 / (1 + last(p)) .* p[1:(end - 1)])
end

@doc raw"""
    convert(::Type{PoincareBallPoint}, p::PoincareHalfSpacePoint)

convert a point [`PoincareHalfSpacePoint`](@ref) `p` (from ``ℝ^n``) from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold ``\mathcal H^n`` to a [`PoincareBallPoint`](@ref) ``π(p) ∈ ℝ^n``.
Denote by ``\tilde p = (p_1,\ldots,p_{d-1})^{\mathrm{T}}``. Then the isometry is defined by

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
    convert(::Type{PoincareBallTangentVector}, p::HyperboloidPoint, X::HyperboloidTangentVector)
    convert(::Type{PoincareBallTangentVector}, p::P, X::T) where {P<:AbstractVector, T<:AbstractVector}

convert a [`HyperboloidTangentVector`](@ref) `X` at `p` to a [`PoincareBallTangentVector`](@ref)
on the [`Hyperbolic`](@ref) manifold ``\mathcal H^n`` by computing the push forward ``π_*(p)[X]`` of
the isometry ``π`` that maps from the Hyperboloid to the Poincaré ball,
cf. [`convert(::Type{PoincareBallPoint}, ::HyperboloidPoint)`](@ref).

The formula reads

````math
π_*(p)[X] = \frac{1}{p_{n+1}+1}\Bigl(\tilde X - \frac{X_{n+1}}{p_{n+1}+1}\tilde p \Bigl),
````

where ``\tilde X = \begin{pmatrix}X_1\\⋮\\X_n\end{pmatrix}``
and ``\tilde p = \begin{pmatrix}p_1\\⋮\\p_n\end{pmatrix}``.
"""
convert(::Type{PoincareBallTangentVector}, ::Any)
function convert(
        t::Type{PoincareBallTangentVector},
        p::HyperboloidPoint,
        X::HyperboloidTangentVector,
    )
    return convert(t, convert(AbstractVector, p), convert(AbstractVector, p, X))
end
function convert(
        ::Type{PoincareBallTangentVector},
        p::P,
        X::T,
    ) where {P <: AbstractVector, T <: AbstractVector}
    return PoincareBallTangentVector(
        1 / (p[end] + 1) .* (X[1:(end - 1)] .- (X[end] / (p[end] + 1) .* p[1:(end - 1)])),
    )
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTangentVector}},
        (p,X)::Tuple{HyperboloidPoint,HyperboloidTangentVector}
    )
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTangentVector}},
        (p, X)::Tuple{P,T},
    ) where {P<:AbstractVector, T <: AbstractVector}

Convert a [`HyperboloidPoint`](@ref) `p` and a [`HyperboloidTangentVector`](@ref) `X`
to a [`PoincareBallPoint`](@ref) and a [`PoincareBallTangentVector`](@ref) simultaneously,
see [`convert(::Type{PoincareBallPoint}, ::HyperboloidPoint)`](@ref) and
[`convert(::Type{PoincareBallTangentVector}, ::HyperboloidPoint, ::HyperboloidTangentVector)`](@ref)
for the formulae.
"""
function convert(
        ::Type{Tuple{PoincareBallPoint, PoincareBallTangentVector}},
        (p, X)::Tuple{HyperboloidPoint, HyperboloidTangentVector},
    )
    return (convert(PoincareBallPoint, p), convert(PoincareBallTangentVector, p, X))
end
function convert(
        ::Type{Tuple{PoincareBallPoint, PoincareBallTangentVector}},
        (p, X)::Tuple{P, T},
    ) where {P <: AbstractVector, T <: AbstractVector}
    return (convert(PoincareBallPoint, p), convert(PoincareBallTangentVector, p, X))
end

@doc raw"""
    convert(
        ::Type{PoincareBallTangentVector},
        p::PoincareHalfSpacePoint,
        X::PoincareHalfSpaceTangentVector
    )

convert a [`PoincareHalfSpaceTangentVector`](@ref) `X` at `p` to a [`PoincareBallTangentVector`](@ref)
on the [`Hyperbolic`](@ref) manifold ``\mathcal H^n`` by computing the push forward ``π_*(p)[X]`` of
the isometry ``π`` that maps from the Poincaré half space to the Poincaré ball,
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
where ``\tilde p = \begin{pmatrix}p_1\\⋮\\p_{n-1}\end{pmatrix}``.
"""
function convert(
        ::Type{PoincareBallTangentVector},
        p::PoincareHalfSpacePoint,
        X::PoincareHalfSpaceTangentVector,
    )
    den = norm(p.value[1:(end - 1)])^2 + (last(p.value) + 1)^2
    scp = dot(p.value, X.value)
    c1 =
        (2 / den .* X.value[1:(end - 1)]) .-
        (4 * (scp + last(X.value)) / (den^2)) .* p.value[1:(end - 1)]
    c2 = 2 * scp / den - 2 * (norm(p.value)^2 - 1) * (scp + last(X.value)) / (den^2)
    return PoincareBallTangentVector(vcat(c1, c2))
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTangentVector}},
        (p,X)::Tuple{HyperboloidPoint,HyperboloidTangentVector}
    )
    convert(
        ::Type{Tuple{PoincareBallPoint,PoincareBallTangentVector}},
        (p, X)::Tuple{T,T},
    ) where {T <: AbstractVector}

Convert a [`PoincareHalfSpacePoint`](@ref) `p` and a [`PoincareHalfSpaceTangentVector`](@ref) `X`
to a [`PoincareBallPoint`](@ref) and a [`PoincareBallTangentVector`](@ref) simultaneously,
see [`convert(::Type{PoincareBallPoint}, ::PoincareHalfSpacePoint)`](@ref) and
[`convert(::Type{PoincareBallTangentVector}, ::PoincareHalfSpacePoint, ::PoincareHalfSpaceTangentVector)`](@ref)
for the formulae.
"""
function convert(
        ::Type{Tuple{PoincareBallPoint, PoincareBallTangentVector}},
        (p, X)::Tuple{PoincareHalfSpacePoint, PoincareHalfSpaceTangentVector},
    )
    return (convert(PoincareBallPoint, p), convert(PoincareBallTangentVector, p, X))
end

@doc raw"""
    distance(::Hyperbolic, p::PoincareBallPoint, q::PoincareBallPoint)

Compute the distance on the [`Hyperbolic`](@ref) manifold ``\mathcal H^n`` represented in the
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

embed(::Hyperbolic, p::PoincareBallPoint) = p.value
embed!(::Hyperbolic, q, p::PoincareBallPoint) = copyto!(q, p.value)
embed(::Hyperbolic, p::PoincareBallPoint, X::PoincareBallTangentVector) = X.value
function embed!(::Hyperbolic, Y, p::PoincareBallPoint, X::PoincareBallTangentVector)
    return copyto!(Y, X.value)
end

function get_embedding(::Hyperbolic{TypeParameter{Tuple{n}}}, ::PoincareBallPoint) where {n}
    return Euclidean(n)
end
function get_embedding(M::Hyperbolic{Tuple{Int}}, ::PoincareBallPoint)
    n = get_parameter(M.size)[1]
    return Euclidean(n; parameter = :field)
end

@doc raw"""
    inner(::Hyperbolic, p::PoincareBallPoint, X::PoincareBallTangentVector, Y::PoincareBallTangentVector)

Compute the inner product in the Poincaré ball model. The formula reads
````math
g_p(X,Y) = \frac{4}{(1-\lVert p \rVert^2)^2}  ⟨X, Y⟩ .
````
"""
function inner(
        ::Hyperbolic,
        p::PoincareBallPoint,
        X::PoincareBallTangentVector,
        Y::PoincareBallTangentVector,
    )
    return 4 / (1 - norm(p.value)^2)^2 * dot(X.value, Y.value)
end

function norm(M::Hyperbolic, p::PoincareBallPoint, X::PoincareBallTangentVector)
    return sqrt(inner(M, p, X, X))
end

@doc raw"""
    project(::Hyperbolic, ::PoincareBallPoint, ::PoincareBallTangentVector)

projection of tangent vectors in the Poincaré ball model is just the identity, since
the tangent space consists of all ``ℝ^n``.
"""
project(::Hyperbolic, ::PoincareBallPoint, ::PoincareBallTangentVector)

function allocate_result(
        ::Hyperbolic,
        ::typeof(project),
        X::PoincareBallTangentVector,
        ::PoincareBallPoint,
    )
    return PoincareBallTangentVector(allocate(X.value))
end

function project!(
        ::Hyperbolic,
        Y::PoincareBallTangentVector,
        ::PoincareBallPoint,
        X::PoincareBallTangentVector,
    )
    return (Y.value .= X.value)
end
