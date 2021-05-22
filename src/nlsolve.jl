
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

function inverse_retract!(M::AbstractManifold, X, p, q, method::NLsolveInverseRetraction; kwargs...)
    X0 = method.X0 === nothing ? zero_vector(M, p) : method.X0
    res = _inverse_retract_nlsolve(
        M,
        p,
        q,
        method.retraction,
        X0,
        method.project_tangent,
        method.project_point,
        method.nlsolve_kwargs;
        kwargs...,
    )
    if !res.f_converged
        @debug res
        throw(OutOfInjectivityRadiusError())
    end
    return copyto!(X, res.zero)
end

function _inverse_retract_nlsolve(
    M::AbstractManifold,
    p,
    q,
    retraction,
    X0,
    project_tangent,
    project_point,
    nlsolve_kwargs;
    kwargs...,
)
    function f!(F, X)
        project_tangent && project!(M, X, p, X)
        retract!(M, F, p, project(M, p, X), retraction; kwargs...)
        project_point && project!(M, q, q)
        F .-= q
        return F
    end
    isdefined(ManifoldsBase, :NLsolve) ||
        @warn "To use NLsolveInverseRetraction, NLsolve must be loaded using `using NLsolve`."
    res = NLsolve.nlsolve(f!, X0; nlsolve_kwargs...)
    return res
end
function inverse_retract!(M::AbstractPowerManifold, X, q, p, m::NLsolveInverseRetraction)
    return inverse_retract!(M, X, q, p, InversePowerRetraction(m))
end
