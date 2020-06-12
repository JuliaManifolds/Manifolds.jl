
"""
    AbstractRiemannianDiffBackend

An abstract type for diff backends. See [`RiemannianONBDiffBackend`](@ref) for
an example.
"""
abstract type AbstractRiemannianDiffBackend end


"""
    differential(f::Any, c::Curve, t::Real, backend::AbstractDiffBackend = rdiff_backend())

Compute the Riemannian differential of a curve of type `c` represented by function `f`
at time `t` using the given backend.
"""
differential(::Any, ::Curve, ::Real, ::AbstractRiemannianDiffBackend)


"""
    gradient(f::Any, rf::RealField, p, backend::AbstractRiemannianDiffBackend = rdiff_backend())

Compute the Riemannian gradient of a real field of type `rf` represented by function `f`
at point `p` using the given backend.
"""
gradient(::Any, ::RealField, ::Any, ::AbstractRiemannianDiffBackend)

"""
    hessian(f::Any, rf::RealField, p, backend::AbstractRiemannianDiffBackend = rdiff_backend())

Compute the Riemannian Hessian of a real field of type `rf` represented by function `f`
at point `p` using the given backend.
"""
hessian(::Any, ::RealField, ::Any, ::AbstractRiemannianDiffBackend)

"""
    jacobian(f::Any, mt::AbstractMap, p, backend::AbstractRiemannianDiffBackend = rdiff_backend())

Compute the Riemannian Jacobian of a map of type `mt` represented by function `f`
at point `p` using the given backend.
"""
jacobian(::Any, ::AbstractMap, ::Any, ::AbstractRiemannianDiffBackend)

function differential!(f::Any, c::Curve, X, t, backend::AbstractRiemannianDiffBackend)
    return copyto!(X, differential(f, c, t, backend))
end

function gradient!(f, ft::RealField, X, p, backend::AbstractRiemannianDiffBackend)
    return copyto!(X, gradient(f, ft, p, backend))
end

"""
    hessian_vector_product(f, ft::RealField, p, X, backend::AbstractRiemannianDiffBackend)


Compute the product of Riemannian Hessian of a real field of type `rf` represented by
function `f` at point `p` and the tangent vector `X` at point `p` using the given backend.
"""
hessian_vector_product(f, ft::RealField, p, X, backend::AbstractRiemannianDiffBackend)

differential(f, c::Curve, p) = differential(f, c, p, rdiff_backend())

differential!(f, c::Curve, X, p) = differential!(f, c, X, p, rdiff_backend())

gradient(f, rf::RealField, p) = gradient(f, rf, p, rdiff_backend())

gradient!(f, rf::RealField, X, p) = gradient!(f, rf, X, p, rdiff_backend())

hessian(f, rf::RealField, p) = hessian(f, rf, p, rdiff_backend())

function hessian_vector_product(f, rf::RealField, p, X)
    return hessian_vector_product(f, rf, p, X, rdiff_backend())
end

jacobian(f, mt::Map, p) = jacobian(f, mt::Map, p, rdiff_backend())

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

function differential(f, c::Curve, t::Real, backend::RiemannianONBDiffBackend)
    M = codomain(c)
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

function differential!(f, c::Curve, X, t::Real, backend::RiemannianONBDiffBackend)
    M = codomain(c)
    p = f(t)
    onb_coords = _derivative(zero(number_eltype(p)), backend.diff_backend) do h
        return get_coordinates(
            M,
            p,
            inverse_retract(M, p, f(t + h), backend.inverse_retraction),
            backend.basis,
        )
    end
    return get_vector!(M, X, p, onb_coords, backend.basis)
end

function gradient(f, ft::RealField, p, backend::RiemannianONBDiffBackend)
    M = domain(ft)
    X = get_coordinates(M, p, zero_tangent_vector(M, p), backend.basis)
    onb_coords = _gradient(X, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return get_vector(M, p, onb_coords, backend.basis)
end

function gradient!(f, ft::RealField, X, p, backend::RiemannianONBDiffBackend)
    M = domain(ft)
    X2 = get_coordinates(M, p, zero_tangent_vector(M, p), backend.basis)
    onb_coords = _gradient(X2, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return get_vector!(M, X, p, onb_coords, backend.basis)
end

"""
    hessian(f, ft::RealField, p, backend::RiemannianONBDiffBackend)

Compute the Riemannian Hessian using the Euclidean Hessian according to Proposition 5.5.4
from [^Absil2008] (generalized to arbitrary retractions).

[^Absil2008]:
    > Absil, P. A., et al. Optimization Algorithms on Matrix Manifolds. 2008.
"""
function hessian(f, ft::RealField, p, backend::RiemannianONBDiffBackend)
    M = domain(ft)
    X = get_coordinates(M, p, zero_tangent_vector(M, p), backend.basis)
    onb_coords = _hessian(X, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return onb_coords
end

function hessian_vector_product(
    f,
    ft::RealField,
    p,
    X,
    backend::RiemannianONBDiffBackend,
)
    M = domain(ft)
    X_zero = get_coordinates(M, p, zero_tangent_vector(M, p), backend.basis)
    X_coords = get_coordinates(M, p, X, backend.basis)
    onb_coords = _hessian_vector_product(X_zero, X_coords, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return get_vector(M, p, onb_coords, backend.basis)
end

function jacobian(f, mt::AbstractMap, p, backend::RiemannianONBDiffBackend)
    M = domain(mt)
    N = codomain(mt)
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
const _current_rdiff_backend = CurrentRiemannianDiffBackend(RiemannianONBDiffBackend(
    diff_backend(),
    ExponentialRetraction(),
    LogarithmicInverseRetraction(),
    DefaultOrthonormalBasis(),
),)

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

function gradient(f, ft::RealField, p, backend::RiemannianProjectionDiffBackend)
    M = domain(ft)
    amb_grad = _gradient(f, p, backend.diff_backend)
    return project(M, p, amb_grad)
end

function gradient!(f, ft::RealField, X, p, backend::RiemannianProjectionDiffBackend)
    M = domain(ft)
    amb_grad = embed(M, p, X)
    _gradient!(f, amb_grad, p, backend.diff_backend)
    return project!(M, X, p, amb_grad)
end
