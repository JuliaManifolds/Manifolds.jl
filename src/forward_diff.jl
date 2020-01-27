
struct ForwardDiffBackend <: AbstractDiffBackend end

function Manifolds._derivative(f::AbstractCurve, x, ::ForwardDiffBackend)
    return ForwardDiff.derivative(f, x)
end

function _gradient(f::AbstractRealField, x, ::ForwardDiffBackend)
    return ForwardDiff.gradient(f, x)
end

function _jacobian(f::AbstractMap, x, ::ForwardDiffBackend)
    return ForwardDiff.jacobian(f, x)
end

push!(_diff_backends, ForwardDiffBackend())
