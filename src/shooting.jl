struct ShootingInverseRetraction{
    T<:Real,
    R<:AbstractRetractionMethod,
    VT<:AbstractVectorTransportMethod,
} <: AbstractInverseRetractionMethod
    max_iterations::Int
    tolerance::T
    num_transport_points::Int
    retraction::R
    vector_transport::VT
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
        max_iterations=method.max_iterations,
        tolerance=method.tolerance,
        num_transport_points=method.num_transport_points,
        retraction=method.retraction,
        vector_transport=method.vector_transport,
    )
    return X
end

function _shooting!(
    M,
    X,
    p,
    q;
    max_iterations,
    tolerance,
    num_transport_points,
    retraction,
    vector_transport,
)
    T = real(Base.promote_eltype(X, p, q))
    ts = range(zero(T), one(T); length=num_transport_points)
    X .= q .- p
    gap = norm(X)
    project!(M, X, p, X)
    gap < tolerance && return X
    rmul!(X, gap / norm(X))
    i = 1
    ΔX = allocate(X)
    ΔXnew = allocate(ΔX)
    retr_tX = allocate_result(M, retract, p, X)
    retr_tX_new = allocate_result(M, retract, p, X)
    while (gap > tolerance) && (i < max_iterations)
        retract!(M, retr_tX, p, X, retraction)
        ΔX .= retr_tX .- q
        gap = norm(ΔX)
        project!(M, ΔX, retr_tX, ΔX)
        rmul!(ΔX, gap / norm(ΔX))
        for t in reverse(ts)[2:(end - 1)]
            retract!(M, retr_tX_new, p, t * X, retraction)
            vector_transport_to!(M, ΔXnew, retr_tX, ΔX, retr_tX_new, vector_transport)
            retr_tX, retr_tX_new = retr_tX_new, retr_tX
            ΔX, ΔXnew = ΔXnew, ΔX
        end
        vector_transport_to!(M, ΔXnew, retr_tX, ΔX, p, vector_transport)
        X .-= ΔXnew
        i += 1
    end
    return X
end
