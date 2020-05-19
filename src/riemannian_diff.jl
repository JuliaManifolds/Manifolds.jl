
"""
    AbstractRiemannianDiffBackend

An abstract type for diff backends. See [`RiemannianONBDiffBackend`](@ref) for
an example.
"""
abstract type AbstractRiemannianDiffBackend end


"""
    r_derivative(f::AbstractCurve, t::Real, backend::AbstractDiffBackend = rdiff_backend()

Compute the Riemannian derivative of a curve `f` at time `t` using the given backend.
"""
r_derivative(::AbstractCurve, ::Real, ::AbstractRiemannianDiffBackend)


"""
    r_gradient(f::AbstractRealField, p, backend::AbstractRiemannianDiffBackend = rdiff_backend())

Compute the Riemannian gradient of a real field `f` at point `p` using the given backend.
"""
r_gradient(::AbstractRealField, ::Any, ::AbstractRiemannianDiffBackend)

"""
    r_hessian(f::AbstractRealField, p, backend::AbstractRiemannianDiffBackend = rdiff_backend())

Compute the Riemannian Hessian of a real field `f` at point `p` using the given backend.
"""
r_hessian(::AbstractRealField, ::Any, ::AbstractRiemannianDiffBackend)

"""
    r_jacobian(f::AbstractMap, p, backend::AbstractRiemannianDiffBackend = rdiff_backend())

Compute the Riemannian Jacobian of a map `f` at point `p` using the given backend.
"""
r_jacobian(::AbstractMap, ::Any, ::AbstractRiemannianDiffBackend)

function r_derivative(f::AbstractCurve, t, backend::AbstractRiemannianDiffBackend)
    error("r_derivative not implemented for curve $(typeof(f)), point $(typeof(t)) and " *
          "backend $(typeof(backend))")
end

function r_derivative!(f::AbstractCurve, X, t, backend::AbstractRiemannianDiffBackend)
    copyto!(X, r_derivative(f, t, backend))
end

function r_gradient(f::AbstractRealField, p, backend::AbstractRiemannianDiffBackend)
    error("r_gradient not implemented for field $(typeof(f)), point $(typeof(p)) and " *
          "backend $(typeof(backend))")
end

function r_gradient!(f::AbstractRealField, X, p, backend::AbstractRiemannianDiffBackend)
    copyto!(X, r_gradient(f, p, backend))
end

function r_hessian(f::AbstractRealField, p, backend::AbstractRiemannianDiffBackend)
    error("r_hessian not implemented for field $(typeof(f)), point $(typeof(p)) and " *
          "backend $(typeof(backend))")
end

function r_jacobian(f::AbstractMap, p, backend::AbstractRiemannianDiffBackend)
    error("r_jacobian not implemented for map $(typeof(f)), point $(typeof(p)) and " *
          "backend $(typeof(backend))")
end

r_derivative(f::AbstractCurve, p) = r_derivative(f, p, rdiff_backend())

r_derivative!(f::AbstractCurve, X, p) = r_derivative!(f, X, p, rdiff_backend())

r_gradient(f::AbstractRealField, p) = r_gradient(f, p, rdiff_backend())

r_gradient!(f::AbstractRealField, X, p) = r_gradient!(f, X, p, rdiff_backend())

r_hessian(f::AbstractRealField, p) = r_hessian(f, p, rdiff_backend())

r_jacobian(f::AbstractMap, p) = r_jacobian(f::AbstractMap, p, rdiff_backend())

"""
    RiemannianONBDiffBackend(
        diff_backend::AbstractDiffBackend
        retraction::AbstractRetractionMethod
        inverse_retraction::AbstractInverseRetractionMethod
        basis::Union{AbstractOrthonormalBasis,CachedBasis{<:AbstractOrthonormalBasis}},
    ) <: AbstractRiemannianDiffBackend

Riemannian differentiation based on differentiation in an [`AbstractOrthonormalBasis`](@ref)
`basis` using specified `retraction`, `inverse_retraction` and using backend `diff_backend`.
"""
struct RiemannianONBDiffBackend{
    TADBackend<:AbstractDiffBackend,
    TRetr<:AbstractRetractionMethod,
    TInvRetr<:AbstractInverseRetractionMethod,
    TBasis<:Union{
        AbstractOrthonormalBasis,
        CachedBasis{𝔽,<:AbstractOrthonormalBasis{𝔽}} where {𝔽},
    },
} <: AbstractRiemannianDiffBackend
    diff_backend::TADBackend
    retraction::TRetr
    inverse_retraction::TInvRetr
    basis::TBasis
end

function r_derivative(f::AbstractCurve, t::Real, backend::RiemannianONBDiffBackend)
    M = codomain(f)
    p = f(t)
    onb_coords = _derivative(zero(number_eltype(p)), backend.diff_backend) do h
        return get_coordinates(
            M,
            p,
            inverse_retract(M, p, f(t + h), backend.inverse_retraction),
            backend.basis,
        )
    end
    return get_vector(M, p, onb_coords, backend.basis)
end

function r_gradient(f::AbstractRealField, p, backend::RiemannianONBDiffBackend)
    M = domain(f)
    X = get_coordinates(M, p, zero_tangent_vector(M, p), backend.basis)
    onb_coords = _gradient(X, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return get_vector(M, p, onb_coords, backend.basis)
end

"""
    r_hessian(f::AbstractRealField, p, backend::RiemannianONBDiffBackend)

Compute the Riemannian Hessian using the Euclidean Hessian according to Proposition 5.5.4
from [^Absil2008] (generalized to arbitrary retractions).

[^Absil2008]:
    > Absil, P. A., et al. Optimization Algorithms on Matrix Manifolds. 2008.
"""
function r_hessian(f::AbstractRealField, p, backend::RiemannianONBDiffBackend)
    M = domain(f)
    X = get_coordinates(M, p, zero_tangent_vector(M, p), backend.basis)
    onb_coords = _hessian(X, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return onb_coords
end

function r_jacobian(f::AbstractMap, p, backend::RiemannianONBDiffBackend)
    M = domain(f)
    N = codomain(f)
    fp = f(p)
    X = get_coordinates(M, p, zero_tangent_vector(M, p), backend.basis)
    onb_coords = _jacobian(X, backend.diff_backend) do Y
        val = f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
        return get_coordinates(
            N,
            fp,
            inverse_retract(N, fp, val, backend.inverse_retraction),
            backend.basis,
        )
    end
    return onb_coords
end

"""
    CurrentRiemannianDiffBackend(backend::AbstractRiemannianDiffBackend)

A mutable struct for storing the current Riemannian differentiation backend in a global
constant [`_current_rdiff_backend`](@ref).

# See also

[`AbstractRiemannianDiffBackend`](@ref), [`rdiff_backend`](@ref), [`rdiff_backend!`]
"""
mutable struct CurrentRiemannianDiffBackend
    backend::AbstractRiemannianDiffBackend
end

"""
    _current_rdiff_backend

The instance of [`CurrentRiemannianDiffBackend`](@ref) that stores the globally default
differentiation backend.
"""
const _current_rdiff_backend = CurrentRiemannianDiffBackend(
    RiemannianONBDiffBackend(
        diff_backend(),
        ExponentialRetraction(),
        LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
    ),
)

"""
    rdiff_backend() -> AbstractRiemannianDiffBackend

Get the current differentiation backend.
"""
rdiff_backend() = _current_rdiff_backend.backend

"""
    rdiff_backend!(backend::AbstractRiemannianDiffBackend)

Set current backend for differentiation to `backend`.
"""
function rdiff_backend!(backend::AbstractRiemannianDiffBackend)
    _current_rdiff_backend.backend = backend
    return backend
end


"""
    RiemannianProjectionDiffBackend(
        diff_backend::AbstractDiffBackend
    ) <: AbstractRiemannianDiffBackend

Riemannian differentiation based on differentiation in the ambient space and projection to
the given manifold. Differentiation in the ambient space is performed using
the backend `diff_backend`.

Only valid for manifolds that are embedded in a special way in the Euclidean space.
See [^Absil2008], Section 3.6.1 for details.

[^Absil2008]:
    > Absil, P. A., et al. Optimization Algorithms on Matrix Manifolds. 2008.
"""
struct RiemannianProjectionDiffBackend{TADBackend<:AbstractDiffBackend} <:
       AbstractRiemannianDiffBackend
    diff_backend::TADBackend
end

function r_gradient(f::AbstractRealField, p, backend::RiemannianProjectionDiffBackend)
    M = domain(f)
    amb_grad = _gradient(f, p, backend.diff_backend)
    return project(M, p, amb_grad)
end

function r_gradient!(f::AbstractRealField, X, p, backend::RiemannianProjectionDiffBackend)
    M = domain(f)
    amb_grad = embed(M, p, X)
    _gradient!(f, amb_grad, p, backend.diff_backend)
    return project!(M, X, p, amb_grad)
end
