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
    T = real(Base.promote_eltype(X, p, q))
    ts = range(zero(T), one(T); length=method.num_transport_points)
    X .= q .- p
    gap = norm(X)
    project!(M, X, p, X)
    gap < method.tolerance && return X
    rmul!(X, gap / norm(X))
    i = 1
    Xˢ = allocate(X)
    Xˢnew = allocate(Xˢ)
    retr_tX = allocate_result(M, retract, p, X)
    retr_tX_new = allocate_result(M, retract, p, X)
    while (gap > method.tolerance) && (i < method.max_iterations)
        retract!(M, retr_tX, p, X, method.retraction)
        Xˢ .= retr_tX .- q
        gap = norm(Xˢ)
        project!(M, Xˢ, retr_tX, Xˢ)
        rmul!(Xˢ, gap / norm(Xˢ))
        for t in reverse(ts)[2:end-1]
            retract!(M, retr_tX_new, p, t * X, method.retraction)
            vector_transport_to!(M, Xˢnew, retr_tX, Xˢ, retr_tX_new, method.vector_transport)
            retr_tX, retr_tX_new = retr_tX_new, retr_tX
            Xˢ, Xˢnew = Xˢnew, Xˢ
        end
        vector_transport_to!(M, Xˢnew, retr_tX, Xˢ, p, method.vector_transport)
        X .-= Xˢnew
        i += 1
    end
    return X
end
