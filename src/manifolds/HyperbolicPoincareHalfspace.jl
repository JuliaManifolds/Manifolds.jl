function check_point(M::Hyperbolic, p::PoincareHalfSpacePoint; kwargs...)
    if !(last(p.value) > 0)
        return DomainError(
            norm(p.value),
            "The point $(p) does not lie on $(M) since its last entry is nonpositive.",
        )
    end
end

function check_size(M::Hyperbolic, p::PoincareHalfSpacePoint)
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
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTangentVector;
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
    convert(::Type{PoincareHalfSpacePoint}, p::PoincareBallPoint)

convert a point [`PoincareBallPoint`](@ref) `p` (from $ℝ^n$) from the
Poincaré ball model of the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ to a [`PoincareHalfSpacePoint`](@ref) $π(p) ∈ ℝ^n$.
Denote by $\tilde p = (p_1,\ldots,p_{n-1})$. Then the isometry is defined by

````math
π(p) = \frac{1}{\lVert \tilde p \rVert^2 - (p_n-1)^2}
\begin{pmatrix}2p_1\\⋮\\2p_{n-1}\\1-\lVert p\rVert^2\end{pmatrix}.
````
"""
function convert(::Type{PoincareHalfSpacePoint}, p::PoincareBallPoint)
    return PoincareHalfSpacePoint(
        1 / (norm(p.value[1:(end - 1)])^2 + (last(p.value) - 1)^2) .*
        vcat(2 .* p.value[1:(end - 1)], 1 - norm(p.value)^2),
    )
end

@doc raw"""
    convert(::Type{PoincareHalfSpacePoint}, p::Hyperboloid)
    convert(::Type{PoincareHalfSpacePoint}, p)

convert a [`HyperboloidPoint`](@ref) or `Vector``p` (from $ℝ^{n+1}$) from the
Hyperboloid model of the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ to a [`PoincareHalfSpacePoint`](@ref) $π(x) ∈ ℝ^{n}$.

This is done in two steps, namely transforming it to a Poincare ball point and from there further on to a PoincareHalfSpacePoint point.
"""
convert(::Type{PoincareHalfSpacePoint}, ::Any)
function convert(t::Type{PoincareHalfSpacePoint}, p::HyperboloidPoint)
    return convert(t, convert(PoincareBallPoint, p))
end
function convert(t::Type{PoincareHalfSpacePoint}, p::T) where {T<:AbstractVector}
    return convert(t, convert(PoincareBallPoint, p))
end

@doc raw"""
    convert(::Type{PoincareHalfSpaceTangentVector}, p::PoincareBallPoint, X::PoincareBallTangentVector)

convert a [`PoincareBallTangentVector`](@ref) `X` at `p` to a [`PoincareHalfSpacePoint`](@ref)
on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ by computing the push forward $π_*(p)[X]$ of
the isometry ``π`` that maps from the Poincaré ball to the Poincaré half space,
cf. [`convert(::Type{PoincareHalfSpacePoint}, ::PoincareBallPoint)`](@ref).

The formula reads

````math
π_*(p)[X] =
\frac{1}{\lVert \tilde p\rVert^2 + (1-p_n)^2}
\begin{pmatrix}
2X_1\\
⋮\\
2X_{n-1}\\
-2⟨X,p⟩
\end{pmatrix}
-
\frac{2}{(\lVert \tilde p\rVert^2 + (1-p_n)^2)^2}
\begin{pmatrix}
2p_1(⟨X,p⟩-X_n)\\
⋮\\
2p_{n-1}(⟨X,p⟩-X_n)\\
(\lVert p \rVert^2-1)(⟨X,p⟩-X_n)
\end{pmatrix}
````
where $\tilde p = \begin{pmatrix}p_1\\⋮\\p_{n-1}\end{pmatrix}$.
"""
function convert(
    ::Type{PoincareHalfSpaceTangentVector},
    p::PoincareBallPoint,
    X::PoincareBallTangentVector,
)
    den = norm(p.value[1:(end - 1)])^2 + (last(p.value) - 1)^2
    scp = dot(p.value, X.value)
    c1 =
        (2 / den .* X.value[1:(end - 1)]) .-
        (4 * (scp - last(X.value)) / (den^2)) .* p.value[1:(end - 1)]
    c2 = -2 * scp / den - 2 * (1 - norm(p.value)^2) * (scp - last(X.value)) / (den^2)
    return PoincareHalfSpaceTangentVector(vcat(c1, c2))
end

@doc raw"""
    convert(::Type{PoincareHalfSpaceTangentVector}, p::HyperboloidPoint, ::HyperboloidTangentVector)
    convert(::Type{PoincareHalfSpaceTangentVector}, p::P, X::T) where {P<:AbstractVector, T<:AbstractVector}

convert a [`HyperboloidTangentVector`](@ref) `X` at `p` to a [`PoincareHalfSpaceTangentVector`](@ref)
on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ by computing the push forward $π_*(p)[X]$ of
the isometry ``π`` that maps from the Hyperboloid to the Poincaré half space,
cf. [`convert(::Type{PoincareHalfSpacePoint}, ::HyperboloidPoint)`](@ref).

This is done similarly to the approach there, i.e. by using the Poincaré ball model as
an intermediate step.
"""
convert(::Type{PoincareHalfSpaceTangentVector}, ::Any)
function convert(
    t::Type{PoincareHalfSpaceTangentVector},
    p::HyperboloidPoint,
    X::HyperboloidTangentVector,
)
    return convert(t, convert(AbstractVector, p), convert(AbstractVector, X))
end
function convert(
    ::Type{PoincareHalfSpaceTangentVector},
    p::P,
    X::T,
) where {P<:AbstractVector,T<:AbstractVector}
    return convert(
        PoincareHalfSpaceTangentVector,
        convert(Tuple{PoincareBallPoint,PoincareBallTangentVector}, (p, X))...,
    )
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTangentVector}},
        (p,X)::Tuple{PoincareBallPoint,PoincareBallTangentVector}
    )

Convert a [`PoincareBallPoint`](@ref) `p` and a [`PoincareBallTangentVector`](@ref) `X`
to a [`PoincareHalfSpacePoint`](@ref) and a [`PoincareHalfSpaceTangentVector`](@ref) simultaneously,
see [`convert(::Type{PoincareHalfSpacePoint}, ::PoincareBallPoint)`](@ref) and
[`convert(::Type{PoincareHalfSpaceTangentVector}, ::PoincareBallPoint,::PoincareBallTangentVector)`](@ref)
for the formulae.
"""
function convert(
    ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTangentVector}},
    (p, X)::Tuple{PoincareBallPoint,PoincareBallTangentVector},
)
    return (
        convert(PoincareHalfSpacePoint, p),
        convert(PoincareHalfSpaceTangentVector, p, X),
    )
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTangentVector}},
        (p,X)::Tuple{HyperboloidPoint,HyperboloidTangentVector}
    )
    convert(
        ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTangentVector}},
        (p, X)::Tuple{P,T},
    ) where {P<:AbstractVector, T <: AbstractVector}

Convert a [`HyperboloidPoint`](@ref) `p` and a [`HyperboloidTangentVector`](@ref) `X`
to a [`PoincareHalfSpacePoint`](@ref) and a [`PoincareHalfSpaceTangentVector`](@ref) simultaneously,
see [`convert(::Type{PoincareHalfSpacePoint}, ::HyperboloidPoint)`](@ref) and
[`convert(::Type{PoincareHalfSpaceTangentVector}, ::Tuple{HyperboloidPoint,HyperboloidTangentVector})`](@ref)
for the formulae.
"""
function convert(
    ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTangentVector}},
    (p, X)::Tuple{HyperboloidPoint,HyperboloidTangentVector},
)
    return (
        convert(PoincareHalfSpacePoint, p),
        convert(PoincareHalfSpaceTangentVector, p, X),
    )
end
function convert(
    ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTangentVector}},
    (p, X)::Tuple{P,T},
) where {P<:AbstractVector,T<:AbstractVector}
    return (
        convert(PoincareHalfSpacePoint, p),
        convert(PoincareHalfSpaceTangentVector, p, X),
    )
end

@doc raw"""
    distance(::Hyperbolic, p::PoincareHalfSpacePoint, q::PoincareHalfSpacePoint)

Compute the distance on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ represented in the
Poincaré half space model. The formula reads

````math
d_{\mathcal H^n}(p,q) = \operatorname{acosh}\Bigl( 1 + \frac{\lVert p - q \rVert^2}{2 p_n q_n} \Bigr)
````
"""
function distance(::Hyperbolic, p::PoincareHalfSpacePoint, q::PoincareHalfSpacePoint)
    return acosh(1 + norm(p.value .- q.value)^2 / (2 * p.value[end] * q.value[end]))
end

embed(::Hyperbolic, p::PoincareHalfSpacePoint) = p.value
embed!(::Hyperbolic, q, p::PoincareHalfSpacePoint) = copyto!(q, p.value)
embed(::Hyperbolic, p::PoincareHalfSpacePoint, X::PoincareHalfSpaceTangentVector) = X.value
function embed!(
    ::Hyperbolic,
    Y,
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTangentVector,
)
    return copyto!(Y, X.value)
end

function get_embedding(
    ::Hyperbolic{TypeParameter{Tuple{n}}},
    ::PoincareHalfSpacePoint,
) where {n}
    return Euclidean(n)
end
function get_embedding(M::Hyperbolic{Tuple{Int}}, ::PoincareHalfSpacePoint)
    n = get_parameter(M.size)[1]
    return Euclidean(n; parameter=:field)
end

function ManifoldsBase.get_embedding_type(::Hyperbolic, ::PoincareHalfSpacePoint)
    return ManifoldsBase.IsometricallyEmbeddedManifoldType(ManifoldsBase.NeedsEmbedding())
end

@doc raw"""
    inner(
        ::Hyperbolic,
        p::PoincareHalfSpacePoint,
        X::PoincareHalfSpaceTangentVector,
        Y::PoincareHalfSpaceTangentVector
    )

Compute the inner product in the Poincaré half space model. The formula reads
````math
g_p(X,Y) = \frac{⟨X,Y⟩}{p_n^2}.
````
"""
function inner(
    ::Hyperbolic,
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTangentVector,
    Y::PoincareHalfSpaceTangentVector,
)
    return dot(X.value, Y.value) / last(p.value)^2
end

function norm(M::Hyperbolic, p::PoincareHalfSpacePoint, X::PoincareHalfSpaceTangentVector)
    return sqrt(inner(M, p, X, X))
end

@doc raw"""
    project(::Hyperbolic, ::PoincareHalfSpacePoint ::PoincareHalfSpaceTangentVector)

projection of tangent vectors in the Poincaré half space model is just the identity, since
the tangent space consists of all $ℝ^n$.
"""
project(::Hyperbolic, ::PoincareHalfSpacePoint::PoincareHalfSpaceTangentVector)

function ManifoldsBase.allocate_result_embedding(
    ::Hyperbolic,
    ::typeof(project),
    X::PoincareHalfSpaceTangentVector,
    ::PoincareHalfSpacePoint,
)
    return PoincareHalfSpaceTangentVector(allocate(X.value))
end

function project!(
    ::Hyperbolic,
    Y::PoincareHalfSpaceTangentVector,
    ::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTangentVector,
)
    return (Y.value .= X.value)
end
