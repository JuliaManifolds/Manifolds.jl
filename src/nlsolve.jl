"""
    ApproximateInverseRetraction{T<:AbstractRetractionMethod} <:
    AbstractInverseRetractionMethod

An inverse retraction method for approximating the inverse of a retraction using `NLsolve`.

# Constructor

    ApproximateInverseRetraction(retraction::AbstractRetractionMethod)

Constructs an approximate inverse retraction for the `retraction` method.
"""
struct ApproximateInverseRetraction{T<:AbstractRetractionMethod} <:
       AbstractInverseRetractionMethod
    retraction::T
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
        X0,
        nlsolve_kwargs,
        kwargs...
    )

Approximate the inverse of the retraction specified by `method.retraction` from `p` with
respect to `q` on the [`Manifold`](@ref) `M` using NLsolve. This inverse retraction is
not guaranteed to succeed and probably will not unless `q` is close to `p` and the initial
guess `X0` is close.

If the solver fails to converge, an [`OutOfInjectivityRadiusError`](@ref) is raised.

# Keywords

- `X0`: initial guess of the inverse retraction, defaulting to [`zero_tangent_vector`](@ref).
- `nlsolve_kwargs`: keyword arguments passed to `NLsolve.nlsolve`.
- `kwargs`: keyword arguments passed to [`retract!`](@ref) for the retraction method.
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
    X0 = zero_tangent_vector(M, p),
    nlsolve_kwargs = NamedTuple(),
    kwargs...,
)
    # TODO: this should probably use Optim and do manifold optimization on the Frobenius norm
    retraction = method.retraction
    function f!(F, X)
        retract!(M, F, p, X, retraction; kwargs...)
        F .-= q
        return F
    end
    res = nlsolve(f!, X0; nlsolve_kwargs...)
    if !res.f_converged
        @debug res
        throw(OutOfInjectivityRadiusError())
    end
    return copyto!(X, res.zero)
end
