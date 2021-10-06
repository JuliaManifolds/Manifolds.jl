
"""
    FiniteDiffBackend <: AbstractDiffBackend

A type to specify / use differentiation backend based on FiniteDiff package.

# Constructor
    FiniteDiffBackend(method::Val{Symbol} = Val{:central})
"""
struct FiniteDiffBackend{TM<:Val} <: AbstractDiffBackend
    method::TM
end

FiniteDiffBackend() = FiniteDiffBackend(Val(:central))

function _derivative(f, p, ::FiniteDiffBackend{Method}) where {Method}
    return FiniteDiff.finite_difference_derivative(f, p, Method)
end

function _gradient(f, p, ::FiniteDiffBackend{Method}) where {Method}
    return FiniteDiff.finite_difference_gradient(f, p, Method)
end

function _gradient!(f, X, p, ::FiniteDiffBackend{Method}) where {Method}
    return FiniteDiff.finite_difference_gradient!(X, f, p, Method)
end

function _jacobian(f, p, ::FiniteDiffBackend{Method}) where {Method}
    return FiniteDiff.finite_difference_jacobian(f, p, Method)
end

function _jacobian!(f, X, p, ::FiniteDiffBackend{Method}) where {Method}
    return FiniteDiff.finite_difference_jacobian!(X, f, p, Method)
end

if default_differential_backend() === NoneDiffBackend()
    set_default_differential_backend!(FiniteDiffBackend())
end
