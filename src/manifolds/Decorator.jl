#
# General functions for decorators
#
"""
    AbstractDecoratorManifold<:Manifold

"""
abstract type AbstractDecoratorManifold<:Manifold end
"""
    is_default_decorator(M)

For any manifold that [`is_decorator_manifold`](@ref) it might be the default
decorator. This function dispatches onto these checks for specific decorators
"""
is_default_decorator(M::Manifold) = false

"""
    is_decorator_transparent(M, f)

Given a [`Manifold`](@ref) `M` and a function `f`, indicate, whether a
[`is_decorator_manifold`](@ref) acts transparent for `f`. This means, it
just passes through down to the internally stored manifold.
Only decorator manifolds can be transparent and their default is, to be transparent.
A function that is affected by the decorator hence has to set this to `Val(false)`
actively.
If a decorator manifold is not in general transparent, it might still pass down
for the case that a decorator is the default decorator, see [`is_default_decorator`](@ref).
"""
is_decorator_transparent(M::Manifold, f) = is_decorator_transparent(M::Manifold, f, is_decorator_manifold(M))
is_decorator_transparent(M::Manifold, f, ::Val{true}) = true
is_decorator_transparent(M::Manifold, f, ::Val{false}) = false


function manifold_function_not_implemented_message(M,f,x...)
    s = join(string.([x...]),", "," and ")
    a = length(x) > 1 ? "arguments" : "argument"
    m = length(x) > 0 ? " for $(a) $(s)." : "."
    return "$(f) not implemented on $(M)$(m)"
end

_acts_transparent(M::Manifold, f) = is_default_decorator(M) || is_decorator_transparent(M,f)

#
# Functions overwritten with decorators
#

function check_manifold_point(M::DT, p; kwargs...) where {DT<:AbstractDecoratorManifold}
    return check_manifold_point(
        M,
        p,
        Val(_acts_transparent(M::Manifold, check_manifold_point));
        kwargs...)
end
function check_manifold_point(M::DT, p,::Val{true}; kwargs...) where {DT<:AbstractDecoratorManifold}
    return check_manifold_point(M.manifold, p; kwargs...)
end
function check_manifold_point(M::DT, p, ::Val{false}; kwargs...) where {DT<:AbstractDecoratorManifold}
    manifold_function_not_implemented_message(M, check_manifold_point, p)
end

function check_tangent_vector(M::DT, p, X; kwargs...) where {DT<:AbstractDecoratorManifold}
    return check_manifold_point(
        M,
        p,
        Val(_acts_transparent(M,check_tangent_vector));
        kwargs...)
end
function check_tangent_vector(M::DT, p, X, ::Val{true}; kwargs...) where {DT<:AbstractDecoratorManifold}
    return check_tangent_vector(M.manifold, p; kwargs...)
end
function check_tangent_vector(M::DT, p, X, ::Val{false}; kwargs...) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M,check_tangent_vector,p,X))
end

function distance(M::DT, p, q) where {DT<:AbstractDecoratorManifold}
    return distance(M, p,q, Val(_acts_transparent(M, distance)))
end
function distance(M::DT, p, q, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return distance(M.manifold, p,q)
end
function distance(M::DT, p, q, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, distance, p, q))
end

function exp!(M::DT, q, p, X) where {DT<:AbstractDecoratorManifold}
    return exp!(M, q, p, X, Val(_acts_transparent(M, exp!)))
end
function exp!(M::DT, q, p, X, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return exp!(M.manifold, q, p, X)
end
function exp!(M::DT, q, p, X, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, exp!, q, p, X))
end

function flat!(M::DT, ξ::CoTFVector, p, X::TFVector) where {DT<:AbstractDecoratorManifold}
    return flat!(M, ξ, p, X,Val(_acts_transparent(M,flat!)))
end
function flat!(M::DT, ξ::CoTFVector, p, X::TFVector, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return flat!(M.manifold, ξ, p, X)
end
function flat!(M::DT, ξ::CoTFVector, p, X::TFVector, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, flat, ξ, p, X))
end

function get_basis(M::DT, p, B::BT) where {DT<:AbstractDecoratorManifold, BT<:AbstractBasis}
    return get_basis(M, p, B, Val(_acts_transparent(M, get_basis)))
end
function get_basis(M::DT, p, B::BT, ::Val{true}) where {DT<:AbstractDecoratorManifold, BT<:AbstractBasis}
    return get_basis(M.manifold, p, B)
end
function get_basis(M::DT, p, B::BT, ::Val{false}) where {DT<:AbstractDecoratorManifold, BT<:AbstractBasis}
    error(manifold_function_not_implemented_message(M, get_basis, p, B))
end

function hat!(M::DT, X, p, Xⁱ) where {DT<:AbstractDecoratorManifold}
    return hat!(M, X, p, Xⁱ, Val(_acts_transparent(M, hat!)))
end
function hat!(M::DT, X, p, Xⁱ, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return hat!(M.manifold, X, p, Xⁱ)
end
function hat!(M::DT, X, p, Xⁱ, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, hat!, p, X))
end

function injectivity_radius(M::DT) where {DT<:AbstractDecoratorManifold}
    return injectivity_radius(M, Val(_acts_transparent(M, injectivity_radius)))
end
function injectivity_radius(M::DT, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return injectivity_radius(M.manifold)
end
function injectivity_radius(M::DT, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, injectivity_radius))
end
function injectivity_radius(M::DT, p) where {DT<:AbstractDecoratorManifold}
    return injectivity_radius(M, p, Val(_acts_transparent(M, injectivity_radius)))
end
function injectivity_radius(M::DT, p, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return injectivity_radius(M.manifold, p)
end
function injectivity_radius(M::DT, p, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, injectivity_radius, p))
end

function inner(M::DT, p, X, Y) where {DT<:AbstractDecoratorManifold}
    return inner(M, p, X, Y, Val(_acts_transparent(M, inner)))
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
    return inverse_retract!(M, X, p, q, m, Val(_acts_transparent(M, inverse_retract!)))
end
function inverse_retract!(
    M::DT,
    X,
    p,
    q,
    m::LogarithmicInverseRetraction
) where {DT<:AbstractDecoratorManifold}
    return inverse_retract!(M, X, p, q, m, Val(_acts_transparent(M, inverse_retract!)))
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
    return isapprox(M, p, q, Val(_acts_transparent(M, isapprox)))
end
function isapprox(M::DT, p, q, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return isapprox(M.manifold, p, q)
end
function isapprox!(M::DT, p, q, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, isapprox, p, q))
end

function isapprox(M::DT, p, X, Y) where {DT<:AbstractDecoratorManifold}
    return isapprox(M, p, X, Y, Val(_acts_transparent(M, isapprox)))
end
function isapprox(M::DT, p, X, Y, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return isapprox(M.manifold, p, X, Y)
end
function isapprox!(M::DT, p, X, Y, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, isapprox, p, X, Y))
end


function log!(M::DT, X, p, q) where {DT<:AbstractDecoratorManifold}
    return log!(M, X, p, q, Val(_acts_transparent(M, log!)))
end
function log!(M::DT, X, p, q, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return log!(M.manifold, X, p, q)
end
function log!(M::DT, X, p, q, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, log!, p, q))
end

function manifold_dimension(M::DT) where {DT<:AbstractDecoratorManifold}
    return manifold_dimension(M, Val(_acts_transparent(M, manifold_dimension)))
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
    return mean!(M, x, w, Val(_acts_transparent(M, mean!)); kwargs...)
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
    return median!(M, x, w, Val(_acts_transparent(M, median!)); kwargs...)
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
        Val(_acts_transparent(M, normal_tvector_distribution))
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
    return project_point!(M, p, Val(_acts_transparent(M, project_point!)))
end
function project_point!(M::DT, p, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return project_point!(M.manifold, p)
end
function project_point!(M::DT, p, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, project_point!, p))
end

function project_tangent!(M::DT, Y, p, X) where {DT<:AbstractDecoratorManifold}
    return project_tangent!(M, Y, p, X, Val(_acts_transparent(M, project_tangent!)))
end
function project_tangent!(M::DT, Y, p, X, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return project_tangent!(M.manifold, Y, p, X)
end
function project_tangent!(M::DT, Y, p, X, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, project_tangent!, Y, p, X))
end

function projected_distribution(M::DT, d, p) where {DT<:AbstractDecoratorManifold}
    return projected_distribution(M, d, p, Val(_acts_transparent(M, projected_distribution)))
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
    return projected_distribution(M, d, Val(_acts_transparent(M, projected_distribution)))
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
    return representation_size(M, Val(_acts_transparent(M, representation_size)))
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
    return retract!(M, q, p, X, m, Val(_acts_transparent(M, retract!)))
end
function retract!(
    M::DT,
    q,
    p,
    X,
    m::ExponentialRetraction
) where {DT<:AbstractDecoratorManifold}
    return retract!(M, q, p, X, m, Val(_acts_transparent(M, retract!)))
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
    return sharp!(M, X, p, ξ, Val(_acts_transparent(M,sharp!)))
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
        Val(_acts_transparent(M,vector_transport_along!))
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
    ::Val{false},
    Y,
    p,
    X,
    c,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, vector_transport_along!, X, p, X, c))
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
        Val(_acts_transparent(M,vector_transport_direction!))
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
    ::Val{false},
    Y,
    p,
    X,
    d,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, vector_transport_direction!, X, p, X, d))
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
        Val(_acts_transparent(M,vector_transport_to!))
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
    ::Val{false},
    Y,
    p,
    X,
    q,
    m,
    ::Val{false}
) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, vector_transport_to!, X, p, X, q, m))
end

function vee!(M::DT, Xⁱ, p, X) where {DT<:AbstractDecoratorManifold}
    return vee!(M, Xⁱ, p, X, Val(_acts_transparent(M,vee!)))
end
function vee!(M::DT, Xⁱ, p, X, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return vee!(M.manifold, Xⁱ, p, X)
end
function vee!(M::DT, Xⁱ, p, X, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, vee!, p, X))
end

function zero_tangent_vector!(M::DT, X, p) where {DT<:AbstractDecoratorManifold}
    return zero_tangent_vector!(M, X, p, Val(_acts_transparent(M,zero_tangent_vector!)))
end
function zero_tangent_vector!(M::DT, X, p, ::Val{true}) where {DT<:AbstractDecoratorManifold}
    return zero_tangent_vector!(M.manifold, X, p)
end
function zero_tangent_vector!(M::DT, X, p, ::Val{false}) where {DT<:AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, zero_tangent_vector!, X, p))
end
