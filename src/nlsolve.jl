function _inverse_retract_nlsolve!(
    M::Manifold,
    X,
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
    res = NLsolve.nlsolve(f!, X0; nlsolve_kwargs...)
    return res
end
