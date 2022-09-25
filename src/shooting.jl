"""
    ShootingInverseRetraction <: ApproximateInverseRetraction

Approximate the inverse of a retraction using the shooting method.

This implementation of the shooting method works by using another inverse retraction to form
the first guess of the vector. This guess is updated by shooting the vector, guessing the
vector pointing from the shooting result to the target point, and transporting this vector
update back to the initial point on aa discretized grid. This process is repeated until the
norm of the vector update falls below a specified tolerance or the maximum number of
iterations is reached.

# Fields
- `retraction::AbstractRetractionMethod`: The retraction whose inverse is to be approximated
- `initial_inverse_retraction::AbstractInverseRetractionMethod`: The inverse retraction used
    to form the initial guess of the vector.
- `vector_transport::AbstractVectorTransportMethod`: The vector transport used to transport
    the initial guess of the vector.
- `num_transport_points::Int`: The number of discretization points used for vector transport
    in the shooting method. 2 is the minimum number of points, including just the endpoints.
- `tolerance::Real`: The tolerance for the shooting method.
- `max_iterations::Int`: The maximum number of iterations for the shooting method.
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
    _shooting!(
        M,
        X,
        p,
        q;
        retraction=method.retraction,
        initial_inverse_retraction=method.initial_inverse_retraction,
        vector_transport=method.vector_transport,
        num_transport_points=method.num_transport_points,
        tolerance=method.tolerance,
        max_iterations=method.max_iterations,
    )
    return X
end

function _shooting!(
    M,
    X,
    p,
    q;
    retraction,
    initial_inverse_retraction,
    vector_transport,
    num_transport_points,
    tolerance,
    max_iterations,
)
    T = real(Base.promote_eltype(X, p, q))
    inverse_retract!(M, X, p, q, initial_inverse_retraction)
    gap = norm(M, p, X)
    gap < tolerance && return X
    ts = range(zero(T), one(T); length=num_transport_points)
    ΔX = allocate(X)
    ΔXnew = allocate(ΔX)
    retr_tX = allocate_result(M, retract, p, X)
    retr_tX_new = allocate_result(M, retract, p, X)
    i = 1
    while (gap > tolerance) && (i < max_iterations)
        retract!(M, retr_tX, p, X, retraction)
        inverse_retract!(M, ΔX, retr_tX, q, initial_inverse_retraction)
        gap = norm(M, retr_tX, ΔX)
        for t in reverse(ts)[2:(end - 1)]
            retract!(M, retr_tX_new, p, t * X, retraction)
            vector_transport_to!(M, ΔXnew, retr_tX, ΔX, retr_tX_new, vector_transport)
            retr_tX, retr_tX_new = retr_tX_new, retr_tX
            ΔX, ΔXnew = ΔXnew, ΔX
        end
        vector_transport_to!(M, ΔXnew, retr_tX, ΔX, p, vector_transport)
        X .+= ΔXnew
        i += 1
    end
    return X
end
