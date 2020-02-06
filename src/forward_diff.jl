
struct ForwardDiffBackend <: AbstractDiffBackend end

function Manifolds._derivative(f::AbstractCurve, p, ::ForwardDiffBackend)
    return ForwardDiff.derivative(f, p)
end

function _gradient(f::AbstractRealField, p, ::ForwardDiffBackend)
    return ForwardDiff.gradient(f, p)
end

function _jacobian(f::AbstractMap, p, ::ForwardDiffBackend)
    return ForwardDiff.jacobian(f, p)
end

push!(_diff_backends, ForwardDiffBackend())
