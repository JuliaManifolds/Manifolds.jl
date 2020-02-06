
"""
    FiniteDiffBackend(method::Val{Symbol} = Val{:central})

Differentiation backend based on FiniteDiff package.
"""
struct FiniteDiffBackend{TM<:Val} <: AbstractDiffBackend
    method::TM
end

FiniteDiffBackend() = FiniteDiffBackend(Val(:central))

push!(_diff_backends, FiniteDiffBackend())

function _derivative(f::AbstractCurve, p, backend::FiniteDiffBackend{Method}) where Method
    return FiniteDiff.finite_difference_derivative(f, p, Method)
end

function _gradient(f::AbstractRealField, p, backend::FiniteDiffBackend{Method}) where Method
    return FiniteDiff.finite_difference_gradient(f, p, Method)
end

function _jacobian(f::AbstractMap, p, backend::FiniteDiffBackend{Method}) where Method
    return FiniteDiff.finite_difference_jacobian(f, p, Method)
end
