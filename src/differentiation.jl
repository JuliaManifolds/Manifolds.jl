
"""
    AbstractDiffBackend


An abstract type for diff backends. See [`FiniteDifferencesBackend`](@ref) for
an example.
"""
abstract type AbstractDiffBackend end

struct NoneDiffBackend <: AbstractDiffBackend end


"""
    _derivative(f, t[, backend::AbstractDiffBackend])

Compute the derivative of a callable `f` at time `t` computed using the given `backend`,
an object of type [`AbstractDiffBackend`](@ref). If the backend is not explicitly
specified, it is obtained using the function [`diff_backend`](@ref).

!!! note

    Not specifying the backend explicitly will usually result in a type instability
    and decreased performance.
"""
function _derivative(f, t, backend::AbstractDiffBackend)
    return error(
        "_derivative not implemented for curve $(typeof(f)), time $(typeof(t)) and " *
        "backend $(typeof(backend))",
    )
end

function _derivative!(f, X, t, backend::AbstractDiffBackend)
    return copyto!(X, _derivative(f, t, backend))
end

"""
    _gradient(f, p[, backend::AbstractDiffBackend])

Compute the gradient of a callable `f` at point `p` computed using the given `backend`,
an object of type [`AbstractDiffBackend`](@ref). If the backend is not explicitly
specified, it is obtained using the function [`diff_backend`](@ref).

!!! note

    Not specifying the backend explicitly will usually result in a type instability
    and decreased performance.
"""
function _gradient(f, p, backend::AbstractDiffBackend)
    return error(
        "_gradient not implemented for field $(typeof(f)), point $(typeof(p)) and " *
        "backend $(typeof(backend))",
    )
end

function _gradient!(f, X, p, backend::AbstractDiffBackend)
    return copyto!(X, _gradient(f, p, backend))
end

"""
    _hessian(f, p[, backend::AbstractDiffBackend])

Compute the Hessian of a callable `f` at point `p` computed using the given `backend`,
an object of type [`AbstractDiffBackend`](@ref). If the backend is not explicitly
specified, it is obtained using the function [`diff_backend`](@ref).

!!! note

    Not specifying the backend explicitly will usually result in a type instability
    and decreased performance.
"""
function _hessian(f, p, backend::AbstractDiffBackend)
    return transpose(_jacobian(q -> _gradient(f, q, backend), p, backend))
end

"""
    _hessian_vector_product(f, p, X[, backend::AbstractDiffBackend])

Compute product of the Hessian of a callable `f` at point `p` computed using the given
`backend`, an object of type [`AbstractDiffBackend`](@ref), and the vector X. If the backend
is not explicitly specified, it is obtained using the function [`diff_backend`](@ref).

!!! note

    Not specifying the backend explicitly will usually result in a type instability
    and decreased performance.
"""
function _hessian_vector_product(f, p, X, backend::AbstractDiffBackend)
    return _jacobian_transpose_vector_product(q -> _gradient(f, q, backend), p, X, backend)
end

"""
    _jacobian(f, p[, backend::AbstractDiffBackend])

Compute the Jacobian of a callable `f` at point `p` computed using the given `backend`,
an object of type [`AbstractDiffBackend`](@ref). If the backend is not explicitly
specified, it is obtained using the function [`diff_backend`](@ref).

!!! note

    Not specifying the backend explicitly will usually result in a type instability
    and decreased performance.
"""
function _jacobian(f, p, backend::AbstractDiffBackend)
    return error(
        "_jacobian not implemented for map $(typeof(f)), point $(typeof(p)) and " *
        "backend $(typeof(backend))",
    )
end

function _jacobian_vector_product(f, p, X, backend::AbstractDiffBackend)
    return _jacobian(f, p, backend) * X
end

function _jacobian_transpose_vector_product(f, p, X, backend::AbstractDiffBackend)
    return transpose(_jacobian(f, p, backend)) * X
end


"""
    CurrentDiffBackend(backend::AbstractDiffBackend)

A mutable struct for storing the current differentiation backend in a global
constant [`_current_diff_backend`](@ref).

# See also

[`AbstractDiffBackend`](@ref), [`diff_backend`](@ref), [`diff_backend!`](@ref)
"""
mutable struct CurrentDiffBackend
    backend::AbstractDiffBackend
end

"""
    _current_diff_backend

The instance of [`CurrentDiffBackend`] that stores the globally default differentiation
backend.
"""
const _current_diff_backend = CurrentDiffBackend(NoneDiffBackend())

"""
    _diff_backends

A vector of valid [`AbstractDiffBackend`](@ref).
"""
const _diff_backends = AbstractDiffBackend[]

"""
    diff_backend() -> AbstractDiffBackend

Get the current differentiation backend.
"""
diff_backend() = _current_diff_backend.backend

"""
    diff_backend!(backend::AbstractDiffBackend)

Set current backend for differentiation to `backend`.
"""
function diff_backend!(backend::AbstractDiffBackend)
    _current_diff_backend.backend = backend
    return backend
end

"""
    diff_backends() -> Vector{AbstractDiffBackend}

Get vector of currently valid differentiation backends.
"""
diff_backends() = _diff_backends

_derivative(f, t) = _derivative(f, t, diff_backend())

_derivative!(f, X, t) = _derivative!(f, X, t, diff_backend())

_gradient(f, p) = _gradient(f, p, diff_backend())

_gradient!(f, X, p) = _gradient!(f, X, p, diff_backend())

_hessian(f, p) = _hessian(f, p, diff_backend())

_hessian_vector_product(f, p, X) = _hessian_vector_product(f, p, X, diff_backend())

_jacobian(f, p) = _jacobian(f, p, diff_backend())

# Finite differences

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

push!(_diff_backends, FiniteDifferencesBackend())

diff_backend!(_diff_backends[end])

function _derivative(f, t, backend::FiniteDifferencesBackend)
    return backend.method(f, t)
end

function _gradient(f, p, backend::FiniteDifferencesBackend)
    return FiniteDifferences.grad(backend.method, f, p)[1]
end

function _jacobian(f, p, backend::FiniteDifferencesBackend)
    return FiniteDifferences.jacobian(backend.method, f, p)[1]
end
