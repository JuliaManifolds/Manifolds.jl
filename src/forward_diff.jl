
struct ForwardDiffBackend <: AbstractDiffBackend end

function Manifolds._derivative(f, p, ::ForwardDiffBackend)
    return ForwardDiff.derivative(f, p)
end

function _derivative!(f, X, t, ::ForwardDiffBackend)
    return ForwardDiff.derivative!(X, f, t)
end

function _gradient(f, p, ::ForwardDiffBackend)
    return ForwardDiff.gradient(f, p)
end

function _gradient!(f, X, t, ::ForwardDiffBackend)
    return ForwardDiff.gradient!(X, f, t)
end

function _jacobian(f, p, ::ForwardDiffBackend)
    return ForwardDiff.jacobian(f, p)
end

push!(_diff_backends, ForwardDiffBackend())
