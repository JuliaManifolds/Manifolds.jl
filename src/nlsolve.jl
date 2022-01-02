
@doc raw"""
    inverse_retract(M, p, q method::NLsolveInverseRetraction; kwargs...)

Approximate the inverse of the retraction specified by `method.retraction` from `p` with
respect to `q` on the [`AbstractManifold`](@ref) `M` using NLsolve. This inverse retraction is
not guaranteed to succeed and probably will not unless `q` is close to `p` and the initial
guess `X0` is close.

If the solver fails to converge, an [`OutOfInjectivityRadiusError`](@ref) is raised.
See [`NLsolveInverseRetraction`](@ref) for configurable parameters.
"""
inverse_retract(::AbstractManifold, p, q, ::NLsolveInverseRetraction; kwargs...)

function inverse_retract_nlsolve!(
    M::AbstractManifold,
    X,
    p,
    q,
    method::NLsolveInverseRetraction;
    kwargs...,
)
    X0 = method.X0 === nothing ? zero_vector(M, p) : method.X0
    res = _inverse_retract_nlsolve(
        M,
        p,
        q,
        m;
        kwargs...,
    )
    return copyto!(X, res.zero)
end

function _inverse_retract_nlsolve(
    M::AbstractManifold,
    p,
    q,
    m;
    kwargs...,
)
    X0 = method.X0 === nothing ? zero_vector(M, p) : method.X0
    function f!(F, X)
        m.project_tangent && project!(M, X, p, X)
        retract!(M, F, p, project(M, p, X), m.retraction; kwargs...)
        m.project_point && project!(M, q, q)
        F .-= q
        return F
    end
    isdefined(Manifolds, :NLsolve) ||
        @warn "To use NLsolveInverseRetraction, NLsolve must be loaded using `using NLsolve`."
    res = NLsolve.nlsolve(f!, X0; m.nlsolve_kwargs...)
    if !res.f_converged
        @debug res
        throw(OutOfInjectivityRadiusError())
    end
    return res
end