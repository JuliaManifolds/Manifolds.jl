function check_manifold_point(
    M::Hyperbolic{N},
    p::PoincareHalfSpacePoint;
    kwargs...,
) where {N}
    mpv = check_manifold_point(Euclidean(N), p.value; kwargs...)
    mpv === nothing || return mpv
    if !(last(p.value) > 0)
        return DomainError(
            norm(p.value),
            "The point $(p) does not lie on $(M) since its last entry is nonpositive.",
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
    convert(::Type{PoincareHalfSpaceTVector}, p::PoincareBallPoint, X::PoincareBallTVector)

convert a [`PoincareBallTVector`](@ref) `X` at `p` to a [`PoincareHalfSpacePoint`](@ref)
on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ by computing the push forward $π_*(p)[X]$ of
the isometry $π$ that maps from the Poincaré ball to the Poincaré half space,
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
    ::Type{PoincareHalfSpaceTVector},
    p::PoincareBallPoint,
    X::PoincareBallTVector,
)
    den = norm(p.value[1:(end - 1)])^2 + (last(p.value) - 1)^2
    scp = dot(p.value, X.value)
    c1 =
        (2 / den .* X.value[1:(end - 1)]) .-
        (4 * (scp - last(X.value)) / (den^2)) .* p.value[1:(end - 1)]
    c2 = -2 * scp / den - 2 * (1 - norm(p.value)^2) * (scp - last(X.value)) / (den^2)
    return PoincareHalfSpaceTVector(vcat(c1, c2))
end

@doc raw"""
    convert(::Type{PoincareHalfSpaceTVector}, p::HyperboloidPoint, ::HyperboloidTVector)
    convert(::Type{PoincareHalfSpaceTVector}, p::P, X::T) where {P<:AbstractVector, T<:AbstractVector}

convert a [`HyperboloidTVector`](@ref) `X` at `p` to a [`PoincareHalfSpaceTVector`](@ref)
on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ by computing the push forward $π_*(p)[X]$ of
the isometry $π$ that maps from the Hyperboloid to the Poincaré half space,
cf. [`convert(::Type{PoincareHalfSpacePoint}, ::HyperboloidPoint)`](@ref).

This is done similarly to the approach there, i.e. by using the Poincaré ball model as
an intermediate step.
"""
convert(::Type{PoincareHalfSpaceTVector}, ::Any)
function convert(
    t::Type{PoincareHalfSpaceTVector},
    p::HyperboloidPoint,
    X::HyperboloidTVector,
)
    return convert(t, convert(AbstractVector, p), convert(AbstractVector, X))
end
function convert(
    ::Type{PoincareHalfSpaceTVector},
    p::P,
    X::T,
) where {P<:AbstractVector,T<:AbstractVector}
    return convert(
        PoincareHalfSpaceTVector,
        convert(Tuple{PoincareBallPoint,PoincareBallTVector}, (p, X))...,
    )
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector}},
        (p,X)::Tuple{PoincareBallPoint,PoincareBallTVector}
    )

Convert a [`PoincareBallPoint`](@ref) `p` and a [`PoincareBallTVector`](@ref) `X`
to a [`PoincareHalfSpacePoint`](@ref) and a [`PoincareHalfSpaceTVector`](@ref) simultaneously,
see [`convert(::Type{PoincareHalfSpacePoint}, ::PoincareBallPoint)`](@ref) and
[`convert(::Type{PoincareHalfSpaceTVector}, ::PoincareBallPoint,::PoincareBallTVector)`](@ref)
for the formulae.
"""
function convert(
    ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector}},
    (p, X)::Tuple{PoincareBallPoint,PoincareBallTVector},
)
    return (convert(PoincareHalfSpacePoint, p), convert(PoincareHalfSpaceTVector, p, X))
end

@doc raw"""
    convert(
        ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector}},
        (p,X)::Tuple{HyperboloidPoint,HyperboloidTVector}
    )
    convert(
        ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector}},
        (p, X)::Tuple{P,T},
    ) where {P<:AbstractVector, T <: AbstractVector}

Convert a [`HyperboloidPoint`](@ref) `p` and a [`HyperboloidTVector`](@ref) `X`
to a [`PoincareHalfSpacePoint`](@ref) and a [`PoincareHalfSpaceTVector`](@ref) simultaneously,
see [`convert(::Type{PoincareHalfSpacePoint}, ::HyperboloidPoint)`](@ref) and
[`convert(::Type{PoincareHalfSpaceTVector}, ::Tuple{HyperboloidPoint,HyperboloidTVector})`](@ref)
for the formulae.
"""
function convert(
    ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector}},
    (p, X)::Tuple{HyperboloidPoint,HyperboloidTVector},
)
    return (convert(PoincareHalfSpacePoint, p), convert(PoincareHalfSpaceTVector, p, X))
end
function convert(
    ::Type{Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector}},
    (p, X)::Tuple{P,T},
) where {P<:AbstractVector,T<:AbstractVector}
    return (convert(PoincareHalfSpacePoint, p), convert(PoincareHalfSpaceTVector, p, X))
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

@doc raw"""
    inner(
        ::Hyperbolic{n},
        p::PoincareHalfSpacePoint,
        X::PoincareHalfSpaceTVector,
        Y::PoincareHalfSpaceTVector
    )

Compute the inner product in the Poincaré half space model. The formula reads
````math
g_p(X,Y) = \frac{⟨X,Y⟩}{p_n^2}.
````
"""
function inner(
    ::Hyperbolic,
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTVector,
    Y::PoincareHalfSpaceTVector,
)
    return dot(X.value, Y.value) / last(p.value)^2
end

@doc raw"""
    project(::Hyperbolic, ::PoincareHalfSpacePoint ::PoincareHalfSpaceTVector)

projction of tangent vectors in the Poincaré half space model is just the identity, since
the tangent space consists of all $ℝ^n$.
"""
project(::Hyperbolic, ::PoincareHalfSpacePoint::PoincareHalfSpaceTVector)

function project!(
    ::Hyperbolic,
    Y::PoincareHalfSpaceTVector,
    ::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTVector,
)
    return (Y.value .= X.value)
end

#
# Plotting Recipe I: points
#
@recipe function f(
    M::Hyperbolic{2},
    pts::AbstractVector{T};
    circle_points = 720,
    geodesic_interpolation = -1,
    hyperbolic_border_color = RGBA(0.0, 0.0, 0.0, 1.0),
) where {T<:PoincareHalfSpacePoint}
    framestyle -> :none
    aspect_ratio --> :equal
    tickfontcolor --> RGBA(1.0, 1.0, 1.0, 1.0)
    if geodesic_interpolation < 0
        seriestype --> :scatter
        return [p.value[1] for p in pts], [p.value[2] for p in pts]
    else
        lpts = empty(pts)
        for i in 1:(length(pts) - 1)
            # push interims points on geodesics between two points.
            push!(
                lpts,
                shortest_geodesic(
                    M,
                    pts[i],
                    pts[i + 1],
                    collect(range(0, 1, length = geodesic_interpolation + 2))[1:(end - 1)], # omit end point
                )...,
            )
        end
        push!(lpts, last(pts)) # add last end point
        # split into x, y, z and plot as curve
        seriestype --> :path
        return [p.value[1] for p in lpts], [p.value[2] for p in lpts]
    end
end
