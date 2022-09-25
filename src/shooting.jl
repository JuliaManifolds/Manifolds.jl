"""
    ShootingInverseRetraction <: ApproximateInverseRetraction

Approximate the inverse of a retraction using [`shooting`](@ref).

# Fields

See [`shooting`](@ref) for descriptions of the following fields:
- `retraction::AbstractRetractionMethod`
- `initial_inverse_retraction::AbstractInverseRetractionMethod`
- `vector_transport::AbstractVectorTransportMethod`
- `num_transport_points::Int`
- `tolerance::Real`
- `max_iterations::Int`
"""
struct ShootingInverseRetraction{
    R<:AbstractRetractionMethod,
    IR<:AbstractInverseRetractionMethod,
    VT<:AbstractVectorTransportMethod,
    T<:Real,
} <: ApproximateInverseRetraction
    retraction::R
    initial_inverse_retraction::IR
    vector_transport::VT
    num_transport_points::Int
    tolerance::T
    max_iterations::Int
end

function inverse_retract(
    M::AbstractDecoratorManifold,
    p,
    q,
    method::ShootingInverseRetraction,
)
    X = allocate_result(M, inverse_retract, p, q)
    inverse_retract!(M, X, p, q, method)
    return X
end
function inverse_retract!(
    M::AbstractDecoratorManifold,
    X,
    p,
    q,
    method::ShootingInverseRetraction,
)
    shooting!(
        M,
        X,
        p,
        q,
        method.retraction;
        initial_inverse_retraction=method.initial_inverse_retraction,
        vector_transport=method.vector_transport,
        num_transport_points=method.num_transport_points,
        tolerance=method.tolerance,
        max_iterations=method.max_iterations,
    )
    return X
end

"""
    shooting(M::AbstractManifold, p, q, retraction::AbstractRetractionMethod; kwargs...)
    shooting!(M::AbstractManifold, X, p, q, retraction::AbstractRetractionMethod; kwargs...)

Approximate the inverse of the `retraction` from `p` to `q` using the shooting method.

In the shooting method, we search for a vector `X` such that
`retract(M, p, X, retraction) ≈ q`.

This implementation of the shooting method works by using another inverse retraction to form
the first guess of the vector. This guess is updated by shooting the vector, guessing the
vector pointing from the shooting result to the target point, and transporting this vector
update back to the initial point on aa discretized grid. This process is repeated until the
norm of the vector update falls below a specified tolerance or the maximum number of
iterations is reached.

# Keywords
- `initial_inverse_retraction::AbstractInverseRetractionMethod`: The inverse retraction used
    to form the initial guess of the vector. Defaults to
    [`default_inverse_retraction`](@ref)`(M)`
- `vector_transport::AbstractVectorTransportMethod`: The vector transport used to transport
    the initial guess of the vector. Defaults to [`default_vector_transport`](@ref)`(M)`.
- `num_transport_points::Int=2`: The number of discretization points used for vector
    transport in the shooting method. 2 is the minimum number of points, including just the
    endpoints.
- `tolerance::Real`: The tolerance for the shooting method. Defaults to `sqrt(eps())` for
    the eltypes of `p` and `q`.
- `max_iterations::Int=10_000`: The maximum number of iterations for the shooting method.
"""
function shooting(
    M::AbstractManifold,
    p,
    q,
    retraction::AbstractRetractionMethod;
    kwargs...,
)
    X = allocate_result(M, inverse_retract, p, q)
    shooting!(M, X, p, q, retraction; kwargs...)
    return X
end
function shooting!(
    M::AbstractManifold,
    X,
    p,
    q,
    retraction::AbstractRetractionMethod;
    initial_inverse_retraction=default_inverse_retraction_method(M),
    vector_transport=default_vector_transport_method(M),
    num_transport_points::Int=2,
    tolerance::Real=sqrt(eps(float(real(eltype(X))))),
    max_iterations::Int=10_000,
)
    inverse_retract!(M, X, p, q, initial_inverse_retraction)
    gap = norm(M, p, X)
    gap < tolerance && return X
    T = real(Base.promote_eltype(X, p, q))
    transport_grid = range(one(T), zero(T); length=num_transport_points)[2:(end - 1)]
    ΔX = allocate(X)
    ΔXnew = tX = allocate(ΔX)
    retr_tX = allocate_result(M, retract, p, X)
    if num_transport_points > 2
        retr_tX_new = allocate_result(M, retract, p, X)
    end
    iteration = 1
    while (gap > tolerance) && (iteration < max_iterations)
        retract!(M, retr_tX, p, X, retraction)
        inverse_retract!(M, ΔX, retr_tX, q, initial_inverse_retraction)
        gap = norm(M, retr_tX, ΔX)
        for t in transport_grid
            tX .= t .* X
            retract!(M, retr_tX_new, p, tX, retraction)
            vector_transport_to!(M, ΔXnew, retr_tX, ΔX, retr_tX_new, vector_transport)
            # realias storage
            retr_tX, retr_tX_new, ΔX, ΔXnew, tX = retr_tX_new, retr_tX, ΔXnew, ΔX, ΔX
        end
        vector_transport_to!(M, ΔXnew, retr_tX, ΔX, p, vector_transport)
        X .+= ΔXnew
        iteration += 1
    end
    return X
end
