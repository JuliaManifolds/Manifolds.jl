
"""
    ForwardDiffBackend <: AbstractDiffBackend

Differentiation backend based on the ForwardDiff.jl package.
"""
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

if default_differential_backend() === NoneDiffBackend()
    set_default_differential_backend!(ForwardDiffBackend())
end
