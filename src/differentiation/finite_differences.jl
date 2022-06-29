"""
    FiniteDifferencesBackend(method::FiniteDifferenceMethod = central_fdm(5, 1))

Differentiation backend based on the FiniteDifferences package.
"""
struct FiniteDifferencesBackend{TM<:FiniteDifferenceMethod} <: AbstractDiffBackend
    method::TM
end

function FiniteDifferencesBackend()
    return FiniteDifferencesBackend(central_fdm(5, 1))
end

function _derivative(f, t, backend::FiniteDifferencesBackend)
    return backend.method(f, t)
end

function _gradient(f, p, backend::FiniteDifferencesBackend)
    return FiniteDifferences.grad(backend.method, f, p)[1]
end

function _jacobian(f, p, backend::FiniteDifferencesBackend)
    return FiniteDifferences.jacobian(backend.method, f, p)[1]
end

function _hessian(f, p, backend::FiniteDifferencesBackend)
    return FiniteDifferences.jacobian(
        backend.method,
        q -> FiniteDifferences.grad(backend.method, f, q)[1],
        p,
    )[1]
end

if default_differential_backend() === NoneDiffBackend()
    set_default_differential_backend!(FiniteDifferencesBackend())
end
