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

function manifold_dimension(::SpecialUnitary{n,𝔽}) where {n,𝔽}
    return manifold_dimension(Unitary(n, 𝔽)) - (real_dimension(𝔽) - 1)
end
