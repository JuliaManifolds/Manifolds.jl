#
# General functions for decorators
#
"""
    AbstractDecoratorManifold <: Manifold

"""
abstract type AbstractDecoratorManifold <: Manifold end
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
is_decorator_transparent(M::Manifold, f) = is_decorator_manifold(M)


function manifold_function_not_implemented_message(M,f,x...)
    s = join(string.([x...]),", "," and ")
    a = length(x) > 1 ? "arguments" : "argument"
    return "$(f) not implemented on $(M) for $(a) $(s)."
end

_acts_transparent(M::Manifold, f) = is_default(M) || is_decorator_transparent(M,f)

#
# Functions overwritten with decorators
#

function check_manifold_point(M::DT, p; kwargs...) where {DT <: AbstractDecoratorManifold}
    return check_manifold_point(
        M,
        Val(_acts_transparent(M::Manifold, check_manifold_point)),
        p;
        kwargs...)
end
function check_manifold_point(M::DT, ::Val{true}, p; kwargs...) where {DT <: AbstractDecoratorManifold}
    return check_manifold_point(M.manifold, p; kwargs...)
end
function check_manifold_point(M::DT, ::Val{false}, p; kwargs...) where {DT <: AbstractDecoratorManifold}
    manifold_function_not_implemented_message(M, check_manifold_point, p)
end

function check_tangent_vector(M::DT, p, X; kwargs...) where {DT <: AbstractDecoratorManifold}
    return check_manifold_point(
        M,
        Val(_acts_transparent(M,check_tangent_vector)),
        p;
        kwargs...)
end
function check_tangent_vector(M::DT, ::Val{true}, p, X; kwargs...) where {DT <: AbstractDecoratorManifold}
    return check_tangent_vector(M.manifold, p; kwargs...)
end
function check_tangent_vector(M::DT, ::Val{false}, p, X; kwargs...) where {DT <: AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M,check_tangent_vector,p,X))
end

function distance(M::DT, p, q) where {DT <: AbstractDecoratorManifold}
    return distance(M, Val(_acts_transparent(M)), p,q)
end
function distance(M::DT, ::Val{true}, p, q) where {DT <: AbstractDecoratorManifold}
    return distance(M.manifold, p,q)
end
function distance(M::DT, p, q, ::Val{false}) where {DT <: AbstractDecoratorManifold}
    error(manifold_function_not_implemented_message(M, distance, p, q))
end

function hat!(M::DT, X, p, Xⁱ) where {DT <: AbstractDecoratorManifold}
    return hat!(M, Val(_acts_transparent(M,hat)), X, p, Xⁱ)
end
function hat!(M::DT, ::Val{true}, X, p, Xⁱ) where {DT <: AbstractDecoratorManifold}
    return hat!(M.manifold, X, p, Xⁱ)
end
function hat!(M::DT, ::Val{false}, X, p, Xⁱ) where {DT <: AbstractDecoratorManifold}
    error("hat! operator not defined for manifold $(typeof(M)), array $(typeof(X)), point $(typeof(p)), and vector $(typeof(Xⁱ))")
end

function vee!(M::DT, Xⁱ, p, X) where {DT <: AbstractDecoratorManifold}
    return vee!(M, Val(_acts_transparent(M,vee)), Xⁱ, p, X)
end
function vee!(M::DT, ::Val{true}, Xⁱ, p, X) where {DT <: AbstractDecoratorManifold}
    return vee!(M.manifold, Xⁱ, p, X)
end
function vee!(M::DT, ::Val{false}, Xⁱ, p, X) where {DT <: AbstractDecoratorManifold}
    error("vee! operator not defined for manifold $(typeof(M)), array $(typeof(X)), point $(typeof(p)), and vector $(typeof(Xⁱ))")
end
