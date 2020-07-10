
"""
    AbstractRiemannianDiffBackend

An abstract type for diff backends. See [`RiemannianONBDiffBackend`](@ref) for
an example.
"""
abstract type AbstractRiemannianDiffBackend end

"""
    differential(f::Any, c::Curve, t::Real, backend::AbstractDiffBackend = rdifferential_backend())

Compute the Riemannian differential of a curve of type `c` represented by function `f`
at time `t` using the given backend.
"""
differential(::Any, ::Curve, ::Real, ::AbstractRiemannianDiffBackend)


"""
    gradient(f::Any, rf::RealField, p, backend::AbstractRiemannianDiffBackend = rgradient_backend())

Compute the Riemannian gradient of a real field of type `rf` represented by function `f`
at point `p` using the given backend.
"""
gradient(::Any, ::RealField, ::Any, ::AbstractRiemannianDiffBackend)

function differential!(f::Any, c::Curve, X, t, backend::AbstractRiemannianDiffBackend)
    return copyto!(X, differential(f, c, t, backend))
end

function gradient!(f, ft::RealField, X, p, backend::AbstractRiemannianDiffBackend)
    return copyto!(X, gradient(f, ft, p, backend))
end

differential(f, c::Curve, p) = differential(f, c, p, rdifferential_backend())

differential!(f, c::Curve, X, p) = differential!(f, c, X, p, rdifferential_backend())

gradient(f, rf::RealField, p) = gradient(f, rf, p, rgradient_backend())

gradient!(f, rf::RealField, X, p) = gradient!(f, rf, X, p, rgradient_backend())

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
    _current_rgradient_backend

The instance of [`CurrentRiemannianGradientBackend`](@ref) that stores the globally default
differentiation backend.
"""
const _current_rgradient_backend = CurrentRiemannianDiffBackend(RiemannianONBDiffBackend(
    diff_backend(),
    ExponentialRetraction(),
    LogarithmicInverseRetraction(),
    DefaultOrthonormalBasis(),
),)

const _current_rdifferential_backend =
    CurrentRiemannianDiffBackend(RiemannianONBDiffBackend(
        diff_backend(),
        ExponentialRetraction(),
        LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
    ),)

"""
    rgradient_backend() -> AbstractRiemannianDiffBackend

Get the current differentiation backend for Riemannian gradients.
"""
rgradient_backend() = _current_rgradient_backend.backend

"""
    rgradient_backend!(backend::AbstractRiemannianDiffBackend)

Set current Riemannian gradient backend for differentiation to `backend`.
"""
function rgradient_backend!(backend::AbstractRiemannianDiffBackend)
    _current_rgradient_backend.backend = backend
    return backend
end

"""
    rdifferential_backend() -> AbstractRiemannianDiffBackend

Get the current differentiation backend for Riemannian differentials.
"""
rdifferential_backend() = _current_rdifferential_backend.backend

"""
    rdifferential_backend!(backend::AbstractRiemannianDiffBackend)

Set current Riemannian differential backend for differentiation to `backend`.
"""
function rdifferential_backend!(backend::AbstractRiemannianDiffBackend)
    _current_rdifferential_backend.backend = backend
    return backend
end

"""
    RiemannianProjectionGradientBackend(
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
struct RiemannianProjectionGradientBackend{TADBackend<:AbstractDiffBackend} <:
       AbstractRiemannianDiffBackend
    diff_backend::TADBackend
end

function gradient(f, ft::RealField, p, backend::RiemannianProjectionGradientBackend)
    M = domain(ft)
    amb_grad = _gradient(f, p, backend.diff_backend)
    return project(M, p, amb_grad)
end

function gradient!(f, ft::RealField, X, p, backend::RiemannianProjectionGradientBackend)
    M = domain(ft)
    amb_grad = embed(M, p, X)
    _gradient!(f, amb_grad, p, backend.diff_backend)
    return project!(M, X, p, amb_grad)
end
