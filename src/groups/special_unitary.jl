struct SpecialUnitary{n,𝔽} <: AbstractEmbeddedManifold{𝔽,TransparentIsometricEmbedding} end

SpecialUnitary(n, 𝔽::AbstractNumbers=ℂ) = SpecialUnitary{n,ℂ}()

function check_manifold_point(G::SpecialUnitary{n,𝔽}, p; kwargs...) where {n,𝔽}
    mpv = check_manifold_point(Euclidean(n, n; field=𝔽), p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(det(p), 1; kwargs...)
        return DomainError(det(p), "The determinant of $p must be +1 but it is $(det(p))")
    end
    if !isapprox(p' * p, one(p); kwargs...)
        return DomainError(
            norm(p' * p - one(p)),
            "$p must be unitary but it's not at kwargs $kwargs",
        )
    end
    return nothing
end

function check_tangent_vector(
    G::SpecialUnitary{n,𝔽},
    p,
    X;
    check_base_point=true,
    kwargs...,
) where {n,𝔽}
    if check_base_point
        mpe = check_manifold_point(G, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = check_tangent_vector(decorated_manifold(G), X; check_base_point=false, kwargs...)
    mpv === nothing || return mpv
    if 𝔽 !== ℝ && !isapprox(tr(X), 0)
        return DomainError(tr(X), "the trace of $X must be 0 but is not at $kwargs")
    end
    return nothing
end

decorated_manifold(::SpecialUnitary{n,𝔽}) where {n,𝔽} = Unitary{n,𝔽}()

decorator_transparent_dispatch(::typeof(project), ::SpecialUnitary, args...) = Val(:parent)

@doc raw"""
    injectivity_radius(G::SpecialUnitary)
    injectivity_radius(G::SpecialUnitary, p)

Return the injectivity radius on the ``\mathrm{SU}(n,𝔽)=``[`SpecialUnitary`](@ref) group
`G`, which is globally ``\sqrt{2} π``.
"""
function injectivity_radius(::SpecialUnitary, p)
    T = float(real(eltype(p)))
    return π * sqrt(T(2))
end
function injectivity_radius(::SpecialUnitary, p, ::ExponentialRetraction)
    T = float(real(eltype(p)))
    return π * sqrt(T(2))
end

function manifold_dimension(::SpecialUnitary{n,𝔽}) where {n,𝔽}
    return manifold_dimension(Unitary(n, 𝔽)) - (real_dimension(𝔽) - 1)
end

"""
    mean(
        G::SpecialUnitary,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/2/√2);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(G::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::SpecialUnitary, ::Any)

function Statistics.mean!(
    G::SpecialUnitary,
    q,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return mean!(G, q, x, w, GeodesicInterpolationWithinRadius(π / 2 / √2); kwargs...)
end

@doc raw"""
    project(G::SpecialUnitary, p)

Project `p` to the nearest point on the [`SpecialUnitary`](@ref) group `G`.

Given the singular value decomposition ``p = U S V^\mathrm{H}``, with the
singular values sorted in descending order, the projection is

````math
\operatorname{proj}_{\mathrm{SU}(n)}(p) =
U\operatorname{diag}\left[1,1,…,\det(U V^\mathrm{H})\right] V^\mathrm{H}.
````

The diagonal matrix ensures that the determinant of the result is $+1$.
"""
project(::SpecialUnitary, ::Any)

function project!(::SpecialUnitary{n}, q, p) where {n}
    F = svd(p)
    detUVt = det(F.U) * det(F.Vt)
    if !isreal(detUVt) || real(detUVt) < 0
        d = similar(F.S, eltype(detUVt))
        fill!(d, 1)
        d[n] = conj(detUVt)
        mul!(q, F.U, Diagonal(d) * F.Vt)
    else
        mul!(q, F.U, F.Vt)
    end
    return q
end

function project!(G::SpecialUnitary{n,𝔽}, Y, p, X) where {n,𝔽}
    inverse_translate_diff!(G, Y, p, p, X, LeftAction())
    project!(SkewHermitianMatrices(n, 𝔽), Y, Y)
    Y[diagind(n, n)] .-= tr(Y) / n
    translate_diff!(G, Y, p, p, Y, LeftAction())
    return Y
end

function Base.show(io::IO, ::SpecialUnitary{n,𝔽}) where {n,𝔽}
    return print(io, "SpecialUnitary($(n), $(𝔽))")
end
