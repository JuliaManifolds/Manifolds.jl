"""
    ApproximateInverseRetraction{T<:AbstractRetractionMethod} <:
    AbstractInverseRetractionMethod

An inverse retraction method for approximating the inverse of a retraction using `NLsolve`.

# Constructor

    ApproximateInverseRetraction(
        method::AbstractRetractionMethod[, X0];
        project_tangent=false,
        project_point=false,
        nlsolve_kwargs...,
    )

Constructs an approximate inverse retraction for the retraction `method` with initial guess
`X0`, defaulting to the zero vector. If `project_tangent` is `true`, then the tangent
vector is projected before the retraction using `project`. If `project_point` is `true`,
then the resulting point is projected after the retraction. `nlsolve_kwargs` are keyword
arguments passed to `NLsolve.nlsolve`.
"""
struct ApproximateInverseRetraction{TR<:AbstractRetractionMethod,TV} <:
       AbstractInverseRetractionMethod
    retraction::TR
    X0::TV
    project_tangent::Bool
    project_point::Bool
    nlsolve_kwargs
end
function ApproximateInverseRetraction(
    m,
    X0=nothing;
    project_tangent::Bool=false,
    project_point::Bool=false,
    nlsolve_kwargs...,
)
    return ApproximateInverseRetraction(
        m,
        X0,
        project_point,
        project_tangent,
        nlsolve_kwargs,
    )
end

@decorator_transparent_signature inverse_retract(
    M::AbstractDecoratorManifold,
    p,
    q,
    m::ApproximateInverseRetraction,
)

@decorator_transparent_signature inverse_retract!(
    M::AbstractDecoratorManifold,
    X,
    p,
    q,
    m::ApproximateInverseRetraction,
)

@doc raw"""
    inverse_retract(
        M,
        p,
        q
        method::ApproximateInverseRetraction;
        kwargs...
    )

Approximate the inverse of the retraction specified by `method.retraction` from `p` with
respect to `q` on the [`Manifold`](@ref) `M` using NLsolve. This inverse retraction is
not guaranteed to succeed and probably will not unless `q` is close to `p` and the initial
guess `X0` is close.

If the solver fails to converge, an [`OutOfInjectivityRadiusError`](@ref) is raised.
See [`ApproximateInverseRetraction`](@ref) for configurable parameters.
"""
function inverse_retract(M::Manifold, p, q, method::ApproximateInverseRetraction; kwargs...)
    X = allocate_result(M, inverse_retract, p, q)
    inverse_retract!(M, X, p, q, method; kwargs...)
    return X
end

function inverse_retract!(
    M::Manifold,
    X,
    p,
    q,
    method::ApproximateInverseRetraction;
    kwargs...,
)
    # TODO: this should probably use Optim and do manifold optimization on the Frobenius norm
    retraction = method.retraction
    project_tangent = method.project_tangent
    project_point = method.project_point
    function f!(F, X)
        project_tangent && project!(M, X, p, X)
        retract!(M, F, p, project(M, p, X), retraction; kwargs...)
        project_point && project!(M, q, q)
        F .-= q
        return F
    end
    X0 = method.X0 === nothing ? zero_tangent_vector(M, p) : method.X0
    res = nlsolve(f!, X0; method.nlsolve_kwargs...)
    if !res.f_converged
        @debug res
        throw(OutOfInjectivityRadiusError())
    end
    return copyto!(X, res.zero)
end
