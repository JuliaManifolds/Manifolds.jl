function Random.rand!(M::AbstractManifold, pX; kwargs...)
    return rand!(Random.default_rng(), M, pX; kwargs...)
end
