"""
    ApproximateInverseRetraction <: AbstractInverseRetractionMethod

An abstract type for representing approximate inverse retraction methods.
"""
abstract type ApproximateInverseRetraction <: AbstractInverseRetractionMethod end

"""
    NLsolveInverseRetraction{T<:AbstractRetractionMethod,TV,TK} <:
        ApproximateInverseRetraction

An inverse retraction method for approximating the inverse of a retraction using `NLsolve`.

# Constructor

    NLsolveInverseRetraction(
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
struct NLsolveInverseRetraction{TR<:AbstractRetractionMethod,TV,TK} <:
       ApproximateInverseRetraction
    retraction::TR
    X0::TV
    project_tangent::Bool
    project_point::Bool
    nlsolve_kwargs::TK
    function NLsolveInverseRetraction(m, X0, project_point, project_tangent, nlsolve_kwargs)
        isdefined(Manifolds, :NLsolve) ||
            @warn "To use NLsolveInverseRetraction, NLsolve must be loaded using `using NLsolve`."
        return new{typeof(m),typeof(X0),typeof(nlsolve_kwargs)}(
            m,
            X0,
            project_point,
            project_tangent,
            nlsolve_kwargs,
        )
    end
end
function NLsolveInverseRetraction(
    m,
    X0=nothing;
    project_tangent::Bool=false,
    project_point::Bool=false,
    nlsolve_kwargs...,
)
    return NLsolveInverseRetraction(m, X0, project_point, project_tangent, nlsolve_kwargs)
end

@decorator_transparent_signature inverse_retract(
    M::AbstractDecoratorManifold,
    p,
    q,
    m::NLsolveInverseRetraction,
)

@decorator_transparent_signature inverse_retract!(
    M::AbstractDecoratorManifold,
    X,
    p,
    q,
    m::NLsolveInverseRetraction,
)

@doc raw"""
    inverse_retract(M, p, q method::NLsolveInverseRetraction; kwargs...)

Approximate the inverse of the retraction specified by `method.retraction` from `p` with
respect to `q` on the [`Manifold`](@ref) `M` using NLsolve. This inverse retraction is
not guaranteed to succeed and probably will not unless `q` is close to `p` and the initial
guess `X0` is close.

If the solver fails to converge, an [`OutOfInjectivityRadiusError`](@ref) is raised.
See [`NLsolveInverseRetraction`](@ref) for configurable parameters.
"""
inverse_retract(::Manifold, p, q, ::NLsolveInverseRetraction; kwargs...)

function inverse_retract!(M::Manifold, X, p, q, method::NLsolveInverseRetraction; kwargs...)
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
