#
# General functions for decorators
#

"""
    AbstractDecoratorManifold <: Manifold
"""
abstract type AbstractDecoratorManifold <: Manifold end

"""
    is_default_decorator(M)

For any manifold that is a subtype of [`AbstractDecoratorManifold`](@ref)
indicates which function is used to determine the default decorator.
This default decorator acts transparent even for functions where transparency is disabled.
"""
is_default_decorator(M::Manifold) = _is_default_decorator(M,val_is_default_decorator(M))
_is_default_decorator(M::Manifold, ::Val{T}) where {T} = T
val_is_default_decorator(M::Manifold) = Val(false)

"""
    is_decorator_transparent(M, f)

Given a [`Manifold`](@ref) `M` and a function `f`, indicate, whether a
[`AbstractDecoratorManifold`](@ref) acts transparent for `f`. This means, it
just passes through down to the internally stored manifold.
Only decorator manifolds can be transparent and their default is, to be transparent.
A function that is affected by the decorator hence has to set this to `Val(false)`
actively.
If a decorator manifold is not in general transparent, it might still pass down
for the case that a decorator is the default decorator, see [`is_default_decorator`](@ref).
"""
is_decorator_transparent(M::Manifold, f) = _is_decorator_transparent(M::Manifold, f, val_is_decorator_transparent(M,f))
_is_decorator_transparent(M::Manifold, f, ::Val{T}) where {T} = T
val_is_decorator_transparent(M::DT, f) where {DT <: Manifold} = Val(true)

function manifold_function_not_implemented_message(M,f,x...)
    s = join(map(string, map(typeof, x)),", "," and ")
    a = length(x) > 1 ? "arguments" : "argument"
    m = length(x) > 0 ? " for $(a) $(s)." : "."
    return "$(f) not implemented on $(M)$(m)"
end

_acts_transparently(M::Manifold, f) = _val_or(val_is_default_decorator(M), val_is_decorator_transparent(M,f))
_val_or(::Val{T1},::Val{T2}) where {T1,T2} = Val(T1||T2)

#
# Functions overwritten with decorators
#

function base_manifold(M::DT, depth=-1) where {DT<: AbstractDecoratorManifold}
    return (depth != 0) ? base_manifold(M.manifold, (depth > 0) ? depth-1 : depth) : M
end
#base_manifold(M::Manifold, depth=-1) = M

function check_manifold_point(M::DT, p; kwargs...) where {DT<:AbstractDecoratorManifold}
    return check_manifold_point(
        M,
        p,
        _acts_transparently(M::Manifold, check_manifold_point);
        kwargs...)
end
function check_manifold_point(M::DT, p,::Val{true}; kwargs...) where {DT<:AbstractDecoratorManifold}
    return check_manifold_point(M.manifold, p; kwargs...)
end
function check_manifold_point(M::DT, p, ::Val{false}; kwargs...) where {DT<:AbstractDecoratorManifold}
    manifold_function_not_implemented_message(M, check_manifold_point, p)
end

function check_tangent_vector(M::DT, p, X; kwargs...) where {DT<:AbstractDecoratorManifold}
    return check_tangent_vector(
        M,
        p,
        X,
        _acts_transparently(M,check_tangent_vector);
        kwargs...)
end
function check_tangent_vector(M::DT, p, X, ::Val{true}; kwargs...) where {DT<:AbstractDecoratorManifold}
    return check_tangent_vector(M.manifold, p, X; kwargs...)
end
function check_tangent_vector(M::DT, p, X, ::Val{false}; kwargs...) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M,check_tangent_vector,p,X))
end

function distance(M::DT, p, q) where {DT<:AbstractDecoratorManifold}
    return distance(M, p,q, _acts_transparently(M, distance))
end
function distance(M::DT, p, q, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return distance(M.manifold, p,q)
end
function distance(M::DT, p, q, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, distance, p, q))
end

function exp!(M::DT, q, p, X) where {DT<:AbstractDecoratorManifold}
    return exp!(M, q, p, X, _acts_transparently(M, exp!))
end
function exp!(M::DT, q, p, X, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return exp!(M.manifold, q, p, X)
end
function exp!(M::DT, q, p, X, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, exp!, q, p, X))
end

function flat!(M::DT, ξ::CoTFVector, p, X::TFVector) where {DT<:AbstractDecoratorManifold}
    return flat!(M, ξ, p, X,_acts_transparently(M,flat!))
end
function flat!(M::DT, ξ::CoTFVector, p, X::TFVector, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return flat!(M.manifold, ξ, p, X)
end
function flat!(M::DT, ξ::CoTFVector, p, X::TFVector, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, flat, ξ, p, X))
end

function get_basis(M::DT, p, B::AbstractBasis) where {DT<:AbstractDecoratorManifold}
    return get_basis(M, p, B, _acts_transparently(M, get_basis))
end
function get_basis(M::DT, p, B::AbstractBasis, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return get_basis(M.manifold, p, B)
end
function get_basis(M::DT, p, B::AbstractBasis, ::Val{false}) where {DT<:AbstractDecoratorManifold, }
    error(manifold_function_not_implemented_message(M, get_basis, p, B))
end

function get_basis(M::DT, p, B::ArbitraryOrthonormalBasis) where {DT<:AbstractDecoratorManifold}
    return get_basis(M, p, B, _acts_transparently(M, get_basis))
end
function get_basis(M::DT, p, B::ArbitraryOrthonormalBasis, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return get_basis(M.manifold, p, B)
end
function get_basis(M::DT, p, B::ArbitraryOrthonormalBasis, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, get_basis, p, B))
end

function get_basis(M::DT, p, B::ProjectedOrthonormalBasis{:svd,ℝ}) where {DT<:AbstractDecoratorManifold}
    return get_basis(M, p, B, _acts_transparently(M, get_basis))
end
function get_basis(M::DT, p, B::ProjectedOrthonormalBasis{:svd,ℝ}, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return get_basis(M.manifold, p, B)
end
function get_basis(M::DT, p, B::ProjectedOrthonormalBasis{:svd,ℝ}, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, get_basis, p, B))
end

function get_coordinates(M::DT, p, X, B::AbstractBasis) where {DT<:AbstractDecoratorManifold}
    return get_coordinates(M, p, X, B, _acts_transparently(M, get_coordinates))
end
function get_coordinates(M::DT, p, X, B::AbstractBasis, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return get_coordinates(M.manifold, p, X, B)
end
function get_coordinates(M::DT, p, X, B::AbstractBasis, ::Val{false}) where {DT<:AbstractDecoratorManifold, }
    error(manifold_function_not_implemented_message(M, get_coordinates, p, X, B))
end

function get_vector(M::DT, p, X, B::AbstractBasis) where {DT<:AbstractDecoratorManifold}
    return get_vector(M, p, X, B, _acts_transparently(M, get_vector))
end
function get_vector(M::DT, p, X, B::AbstractBasis, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return get_vector(M.manifold, p, X, B)
end
function get_vector(M::DT, p, X, B::AbstractBasis, ::Val{false}) where {DT<:AbstractDecoratorManifold, }
    error(manifold_function_not_implemented_message(M, get_vector, p, X, B))
end


function hat!(M::DT, X, p, Xⁱ) where {DT<:AbstractDecoratorManifold}
    return hat!(M, X, p, Xⁱ, _acts_transparently(M, hat!))
end
function hat!(M::DT, X, p, Xⁱ, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return hat!(M.manifold, X, p, Xⁱ)
end
function hat!(M::DT, X, p, Xⁱ, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, hat!, p, X))
end

function injectivity_radius(M::DT) where {DT<:AbstractDecoratorManifold}
    return injectivity_radius(M, _acts_transparently(M, injectivity_radius))
end
function injectivity_radius(M::DT, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return injectivity_radius(M.manifold)
end
function injectivity_radius(M::DT, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, injectivity_radius))
end
function injectivity_radius(M::DT, p) where {DT<:AbstractDecoratorManifold}
    return injectivity_radius(M, p, _acts_transparently(M, injectivity_radius))
end
function injectivity_radius(M::DT, p, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return injectivity_radius(M.manifold, p)
end
function injectivity_radius(M::DT, p, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, injectivity_radius, p))
end

function inner(M::DT, p, X, Y) where {DT<:AbstractDecoratorManifold}
    return inner(M, p, X, Y, _acts_transparently(M, inner))
end
function inner(M::DT, p, X, Y, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return inner(M.manifold, p, X, Y)
end
function inner(M::DT, p, X, Y, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, inner, p, X, Y))
end

function inverse_retract!(
    M::DT,
    X,
    p,
    q,
    m::AbstractInverseRetractionMethod
) where {DT<:AbstractDecoratorManifold}
    return inverse_retract!(M, X, p, q, m, _acts_transparently(M, inverse_retract!))
end
function inverse_retract!(
    M::DT,
    X,
    p,
    q,
    m::LogarithmicInverseRetraction
) where {DT<:AbstractDecoratorManifold}
    return inverse_retract!(M, X, p, q, m, _acts_transparently(M, inverse_retract!))
end
function inverse_retract!(
    M::DT,
    X,
    p,
    q,
    m::AbstractInverseRetractionMethod,
    ::Val{true}
) where {DT<:AbstractDecoratorManifold}
    return inverse_retract!(M.manifold, X, p, q, m)
end
function inverse_retract!(
    M::DT,
    X,
    p,
    q,
    m::AbstractInverseRetractionMethod,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, inverse_retract!, X, p, q, m))
end

function isapprox(M::DT, p, q) where {DT<:AbstractDecoratorManifold}
    return isapprox(M, p, q, _acts_transparently(M, isapprox))
end
function isapprox(M::DT, p, q, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return isapprox(M.manifold, p, q)
end
function isapprox(M::DT, p, q, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, isapprox, p, q))
end

function isapprox(M::DT, p, X, Y) where {DT<:AbstractDecoratorManifold}
    return isapprox(M, p, X, Y, _acts_transparently(M, isapprox))
end
function isapprox(M::DT, p, X, Y, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return isapprox(M.manifold, p, X, Y)
end
function isapprox(M::DT, p, X, Y, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, isapprox, p, X, Y))
end


function log!(M::DT, X, p, q) where {DT<:AbstractDecoratorManifold}
    return log!(M, X, p, q, _acts_transparently(M, log!))
end
function log!(M::DT, X, p, q, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return log!(M.manifold, X, p, q)
end
function log!(M::DT, X, p, q, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, log!, p, q))
end

function manifold_dimension(M::DT) where {DT<:AbstractDecoratorManifold}
    return manifold_dimension(M, _acts_transparently(M, manifold_dimension))
end
function manifold_dimension(M::DT, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return manifold_dimension(M.manifold)
end
function manifold_dimension(M::DT, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, manifold_dimension))
end


function mean!(
    M::DT,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {DT<:AbstractDecoratorManifold}
    return mean!(M, x, w, _acts_transparently(M, mean!); kwargs...)
end
function mean!(
        M::DT,
        x::AbstractVector,
        w::AbstractVector,
        ::Val{true};
        kwargs...
) where {DT<:AbstractDecoratorManifold}
    return mean!(M.manifold, x, w; kawrgs...)
end
function mean!(
        M::DT,
        x::AbstractVector,
        w::AbstractVector,
        ::Val{false};
        kwargs...
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, mean!, x, w))
end

function median!(
    M::DT,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...
) where {DT<:AbstractDecoratorManifold}
    return median!(M, x, w, _acts_transparently(M, median!); kwargs...)
end
function median!(
        M::DT,
        x::AbstractVector,
        w::AbstractVector,
        ::Val{true};
        kwargs...
) where {DT<:AbstractDecoratorManifold}
    return median!(M.manifold, x, w; kawrgs...)
end
function median!(
        M::DT,
        x::AbstractVector,
        w::AbstractVector,
        ::Val{false};
        kwargs...
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, median!, x, w))
end

function normal_tvector_distribution(M::DT, p, σ) where {DT<:AbstractDecoratorManifold}
    return normal_tvector_distribution(
        M,
        p,
        σ,
        _acts_transparently(M, normal_tvector_distribution)
    )
end
function normal_tvector_distribution(
    M::DT,
    p,
    σ,
    ::Val{true}
) where {DT<:AbstractDecoratorManifold}
    return normal_tvector_distribution(M.manifold, p, σ)
end
function normal_tvector_distribution(
    M::DT,
    p,
    σ,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, normal_tvector_distribution, p, σ))
end

function project_point!(M::DT, p) where {DT<:AbstractDecoratorManifold}
    return project_point!(M, p, _acts_transparently(M, project_point!))
end
function project_point!(M::DT, p, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return project_point!(M.manifold, p)
end
function project_point!(M::DT, p, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, project_point!, p))
end

function project_tangent!(M::DT, Y, p, X) where {DT<:AbstractDecoratorManifold}
    return project_tangent!(M, Y, p, X, _acts_transparently(M, project_tangent!))
end
function project_tangent!(M::DT, Y, p, X, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return project_tangent!(M.manifold, Y, p, X)
end
function project_tangent!(M::DT, Y, p, X, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, project_tangent!, Y, p, X))
end

function projected_distribution(M::DT, d, p) where {DT<:AbstractDecoratorManifold}
    return projected_distribution(M, d, p, _acts_transparently(M, projected_distribution))
end
function projected_distribution(M::DT, d, p, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return projected_distribution(M.manifold, d, p)
end
function projected_distribution(
    M::DT,
    d,
    p,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, projected_distribution, d, p))
end
function projected_distribution(M::DT, d) where {DT<:AbstractDecoratorManifold}
    return projected_distribution(M, d, _acts_transparently(M, projected_distribution))
end
function projected_distribution(M::DT, d, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return projected_distribution(M.manifold, d)
end
function projected_distribution(
    M::DT,
    d,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, projected_distribution, d))
end

function representation_size(M::DT) where {DT<:AbstractDecoratorManifold}
    return representation_size(M, _acts_transparently(M, representation_size))
end
function representation_size(M::DT, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return representation_size(M.manifold)
end
function representation_size(M::DT, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, representation_size))
end


function retract!(
    M::DT,
    q,
    p,
    X,
    m::AbstractRetractionMethod
) where {DT<:AbstractDecoratorManifold}
    return retract!(M, q, p, X, m, _acts_transparently(M, retract!))
end
function retract!(
    M::DT,
    q,
    p,
    X,
    m::ExponentialRetraction
) where {DT<:AbstractDecoratorManifold}
    return retract!(M, q, p, X, m, _acts_transparently(M, retract!))
end
function retract!(
    M::DT,
    q,
    p,
    X,
    m::AbstractRetractionMethod,
    ::Val{true}
) where {DT<:AbstractDecoratorManifold}
    return retract!(M.manifold, q, p, X)
end
function retract!(
    M::DT,
    q,
    p,
    X,
    m::AbstractRetractionMethod,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, retract!, q, p, X, m))
end


function sharp!(M::DT, X::TFVector, p, ξ::CoTFVector) where {DT<:AbstractDecoratorManifold}
    return sharp!(M, X, p, ξ, _acts_transparently(M,sharp!))
end
function sharp!(M::DT, X::TFVector, p, ξ::CoTFVector, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return sharp!(M.manifold, X, p, ξ)
end
function sharp!(M::DT, X::TFVector, p, ξ::CoTFVector, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, sharp!, ξ, p, X))
end

function vector_transport_along!(
    M::DT,
    Y,
    p,
    X,
    c
) where {DT<:AbstractDecoratorManifold}
    return vector_transport_along!(
        M,
        Y,
        p,
        X,
        c,
        _acts_transparently(M,vector_transport_along!)
    )
end
function vector_transport_along!(
    M::DT,
    Y,
    p,
    X,
    c,
    ::Val{true}
) where {DT<:AbstractDecoratorManifold}
    return vector_transport_along!(M.manifold, Y, p, X, c)
end
function vector_transport_along!(
    M::DT,
    Y,
    p,
    X,
    c,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, vector_transport_along!, Y, p, X, c))
end


function vector_transport_direction!(
    M::DT,
    Y,
    p,
    X,
    d
) where {DT<:AbstractDecoratorManifold}
    return vector_transport_direction!(
        M,
        Y,
        p,
        X,
        d,
        _acts_transparently(M,vector_transport_direction!)
    )
end
function vector_transport_direction!(
    M::DT,
    Y,
    p,
    X,
    d,
    ::Val{true}
) where {DT<:AbstractDecoratorManifold}
    return vector_transport_direction!(M.manifold, Y, p, X, d)
end
function vector_transport_direction!(
    M::DT,
    Y,
    p,
    X,
    d,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, vector_transport_direction!, Y, p, X, d))
end

function vector_transport_to!(
    M::DT,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
) where {DT<:AbstractDecoratorManifold}
    return vector_transport_to!(
        M,
        Y,
        p,
        X,
        q,
        m,
        _acts_transparently(M,vector_transport_to!)
    )
end
function vector_transport_to!(
    M::DT,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
    ::Val{true}
) where {DT<:AbstractDecoratorManifold}
    return vector_transport_to!(M.manifold, Y, p, X, q, m)
end
function vector_transport_to!(
    M::DT,
    Y,
    p,
    X,
    q,
    m,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, vector_transport_to!, Y, p, X, q, m))
end

function vee!(M::DT, Xⁱ, p, X) where {DT<:AbstractDecoratorManifold}
    return vee!(M, Xⁱ, p, X, _acts_transparently(M,vee!))
end
function vee!(M::DT, Xⁱ, p, X, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return vee!(M.manifold, Xⁱ, p, X)
end
function vee!(M::DT, Xⁱ, p, X, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, vee!, p, X))
end

function zero_tangent_vector!(M::DT, X, p) where {DT<:AbstractDecoratorManifold}
    return zero_tangent_vector!(M, X, p, _acts_transparently(M,zero_tangent_vector!))
end
function zero_tangent_vector!(M::DT, X, p, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return zero_tangent_vector!(M.manifold, X, p)
end
function zero_tangent_vector!(M::DT, X, p, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, zero_tangent_vector!, X, p))
end
