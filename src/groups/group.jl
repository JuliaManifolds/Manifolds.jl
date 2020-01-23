@doc doc"""
    AbstractGroupOperation

Abstract type for smooth binary operations $∘$ on elements of a Lie group $G$:
```math
∘: G × G → G
```
An operation can be either defined for a specific [`AbstractGroupManifold`](@ref)
or in general, by defining for an operation `Op` the following methods:

    identity!(::AbstractGroupManifold{Op}, y, x)
    identity(::AbstractGroupManifold{Op}, x)
    inv!(::AbstractGroupManifold{Op}, y, x)
    inv(::AbstractGroupManifold{Op}, x)
    compose(::AbstractGroupManifold{Op}, x, y)
    compose!(::AbstractGroupManifold{Op}, z, x, y)

Note that a manifold is connected with an operation by wrapping it with a decorator,
[`AbstractGroupManifold`](@ref). In typical cases the concrete wrapper
[`GroupManifold`](@ref) can be used.
"""
abstract type AbstractGroupOperation end

@doc doc"""
    AbstractGroupManifold{<:AbstractGroupOperation} <: Manifold

Abstract type for a Lie group, a group that is also a smooth manifold with an
[`AbstractGroupOperation`](@ref), a smooth binary operation. `AbstractGroupManifold`s must
implement at least [`inv`](@ref), [`identity`](@ref), [`compose`](@ref), and
[`translate_diff`](@ref).
"""
abstract type AbstractGroupManifold{O<:AbstractGroupOperation} <: Manifold end

"""
    GroupManifold{M<:Manifold,O<:AbstractGroupOperation} <: AbstractGroupManifold{O}

Decorator for a smooth manifold that equips the manifold with a group operation, thus making
it a Lie group. See [`AbstractGroupManifold`](@ref) for more details.

Group manifolds by default forward metric-related operations to the wrapped manifold.

# Constructor

    GroupManifold(manifold, op)
"""
struct GroupManifold{M<:Manifold,O<:AbstractGroupOperation} <: AbstractGroupManifold{O}
    manifold::M
    op::O
end

show(io::IO, G::GroupManifold) = print(io, "GroupManifold($(G.manifold), $(G.op))")

is_decorator_manifold(::GroupManifold) = Val(true)

is_decorator_group(::AbstractGroupManifold) = Val(true)
is_decorator_group(M::Manifold) = is_decorator_group(M, is_decorator_manifold(M))
is_decorator_group(M::Manifold, ::Val{true}) = is_decorator_group(M.manifold)
is_decorator_group(::Manifold, ::Val{false}) = Val(false)

"""
    base_group(M::Manifold) -> AbstractGroupManifold

Undecorate `M` until an `AbstractGroupManifold` is encountered.
Return an error if the [`base_manifold`](@ref) is reached without encountering a group.
"""
function base_group(M::Manifold)
    is_decorator_group(M) === Val(true) && return base_group(M.manifold)
    error("base_group: manifold $(typeof(M)) with base manifold $(typeof(base_manifold(M))) has no base group.")
end
base_group(G::AbstractGroupManifold) = G

# piping syntax for decoration
if VERSION ≥ v"1.3"
    (op::AbstractGroupOperation)(M::Manifold) = GroupManifold(M, op)
    (::Type{T})(M::Manifold) where {T<:AbstractGroupOperation} = GroupManifold(M, T())
end

########################
# GroupManifold forwards
########################

function check_tangent_vector(G::GroupManifold, x, v; kwargs...)
    return check_tangent_vector(G.manifold, x, v; kwargs...)
end
distance(G::GroupManifold, x, y) = distance(G.manifold, x, y)
exp(G::GroupManifold, x, v) = exp(G.manifold, x, v)
exp!(G::GroupManifold, y, x, v) = exp!(G.manifold, y, x, v)
injectivity_radius(G::GroupManifold) = injectivity_radius(G.manifold)
injectivity_radius(G::GroupManifold, x) = injectivity_radius(G.manifold, x)
function injectivity_radius(G::GroupManifold, x, method::AbstractRetractionMethod)
    return injectivity_radius(G.manifold, x, method)
end
inner(G::GroupManifold, x, v, w) = inner(G.manifold, x, v, w)
inverse_retract(G::GroupManifold, x, y) = inverse_retract(G.manifold, x, y)
function inverse_retract(G::GroupManifold, x, y, method::AbstractInverseRetractionMethod)
    return inverse_retract(G.manifold, x, y, method)
end
inverse_retract!(G::GroupManifold, v, x, y) = inverse_retract!(G.manifold, v, x, y)
function inverse_retract!(
    G::GroupManifold,
    v,
    x,
    y,
    method::AbstractInverseRetractionMethod,
)
    return inverse_retract!(G.manifold, v, x, y, method)
end
function inverse_retract!(G::GroupManifold, v, x, y, method::LogarithmicInverseRetraction)
    return inverse_retract!(G.manifold, v, x, y, method)
end
isapprox(G::GroupManifold, x, y; kwargs...) = isapprox(G.manifold, x, y; kwargs...)
isapprox(G::GroupManifold, x, v, w; kwargs...) = isapprox(G.manifold, x, v, w; kwargs...)
log(G::GroupManifold, x, y) = log(G.manifold, x, y)
log!(G::GroupManifold, v, x, y) = log!(G.manifold, v, x, y)
norm(G::GroupManifold, x, v) = norm(G.manifold, x, v)
project_point(G::GroupManifold, x) = project_point(G.manifold, x)
project_point!(G::GroupManifold, y, x) = project_point!(G.manifold, y, x)
project_tangent(G::GroupManifold, x, v) = project_tangent(G.manifold, x, v)
project_tangent!(G::GroupManifold, w, x, v) = project_tangent!(G.manifold, w, x, v)
retract(G::GroupManifold, x, v) = retract(G.manifold, x, v)
function retract(G::GroupManifold, x, v, method::AbstractRetractionMethod)
    return retract(G.manifold, x, v, method)
end
retract!(G::GroupManifold, y, x, v) = retract!(G.manifold, y, x, v)
function retract!(G::GroupManifold, y, x, v, method::AbstractRetractionMethod)
    return retract!(G.manifold, y, x, v, method)
end
function retract!(G::GroupManifold, y, x, v, method::ExponentialRetraction)
    return retract!(G.manifold, y, x, v, method)
end
function vector_transport_along!(G::GroupManifold, vto, x, v, c, args...)
    return vector_transport_along!(G.manifold, vto, x, v, c, args...)
end
function vector_transport_along(G::GroupManifold, x, v, c, args...)
    return vector_transport_along(G.manifold, x, v, c, args...)
end
function vector_transport_direction!(G::GroupManifold, vto, x, v, vdir, args...)
    return vector_transport_direction!(G.manifold, vto, x, v, vdir, args...)
end
function vector_transport_direction(G::GroupManifold, x, v, vdir, args...)
    return vector_transport_direction(G.manifold, x, v, vdir, args...)
end
function vector_transport_to!(G::GroupManifold, vto, x, v, y, args...)
    return vector_transport_to!(G.manifold, vto, x, v, y, args...)
end
function vector_transport_to(G::GroupManifold, x, v, y, args...)
    return vector_transport_to(G.manifold, x, v, y, args...)
end
zero_tangent_vector(G::GroupManifold, x) = zero_tangent_vector(G.manifold, x)
zero_tangent_vector!(G::GroupManifold, y, x) = zero_tangent_vector!(G.manifold, y, x)

###################
# Action directions
###################

"""
    ActionDirection

Direction of action on a manifold, either [`LeftAction`](@ref) or [`RightAction`](@ref).
"""
abstract type ActionDirection end

"""
    LeftAction()

Left action of a group on a manifold.
"""
struct LeftAction <: ActionDirection end

"""
    RightAction()

Right action of a group on a manifold.
"""
struct RightAction <: ActionDirection end

"""
    switch_direction(::ActionDirection)

Returns a [`RightAction`](@ref) when given a [`LeftAction`](@ref) and vice versa.
"""
switch_direction(::ActionDirection)
switch_direction(::LeftAction) = RightAction()
switch_direction(::RightAction) = LeftAction()

##################################
# General Identity element methods
##################################

@doc doc"""
    Identity(G::AbstractGroupManifold)

The group identity element $e ∈ G$.
"""
struct Identity{G<:AbstractGroupManifold}
    group::G
end

Identity(M::Manifold) = Identity(M, is_decorator_manifold(M))
Identity(M::Manifold, ::Val{true}) = Identity(M.manifold)
Identity(M::Manifold, ::Val{false}) = error("Identity not implemented for manifold $(M)")

show(io::IO, e::Identity) = print(io, "Identity($(e.group))")

(e::Identity)(x) = identity(e.group, x)

# To ensure similar_result_type works
eltype(e::Identity) = Bool

copyto!(e::TE, ::TE) where {TE<:Identity} = e
copyto!(x, ::TE) where {TE<:Identity} = identity!(e.group, x, e)
copyto!(x::AbstractArray, e::TE) where {TE<:Identity} = identity!(e.group, x, e)

isapprox(x, e::Identity; kwargs...) = isapprox(e::Identity, x; kwargs...)
isapprox(e::Identity, x; kwargs...) = isapprox(e.group, e, x; kwargs...)
isapprox(e::E, ::E; kwargs...) where {E<:Identity} = true

function check_manifold_point(M::Manifold, x::Identity; kwargs...)
    if is_decorator_group(M) === Val(true)
        return check_manifold_point(base_group(M), x; kwargs...)
    end
    return DomainError(x, "The identity element $(x) does not belong to $(M).")
end
function check_manifold_point(G::GroupManifold, x::Identity; kwargs...)
    x === Identity(G) && return nothing
    return DomainError(x, "The identity element $(x) does not belong to $(G).")
end
function check_manifold_point(G::GroupManifold, x; kwargs...)
    return check_manifold_point(G.manifold, x; kwargs...)
end

##########################
# Group-specific functions
##########################

@doc doc"""
    inv(G::AbstractGroupManifold, x)

Inverse $x^{-1} ∈ G$ of an element $x ∈ G$, such that
$x \circ x^{-1} = x^{-1} \circ x = e ∈ G$.
"""
inv(M::Manifold, x) = inv(M, x, is_decorator_manifold(M))
inv(M::Manifold, x, ::Val{true}) = inv(M.manifold, x)
function inv(M::Manifold, x, ::Val{false})
    return error("inv not implemented on $(typeof(M)) for points $(typeof(x))")
end
function inv(G::AbstractGroupManifold, x)
    y = similar_result(G, inv, x)
    return inv!(G, y, x)
end

@doc doc"""
    inv!(G::AbstractGroupManifold, y, x)

Inverse $x^{-1} ∈ G$ of an element $x ∈ G$, such that
$x \circ x^{-1} = x^{-1} \circ x = e ∈ G$.
The result is saved to `y`.
"""
inv!(M::Manifold, y, x) = inv!(M, y, x, is_decorator_manifold(M))
inv!(M::Manifold, y, x, ::Val{true}) = inv!(M.manifold, y, x)
function inv!(M::Manifold, y, x, ::Val{false})
    return error("inv! not implemented on $(typeof(M)) for points $(typeof(x))")
end

@doc doc"""
    identity(G::AbstractGroupManifold, x)

Identity element $e ∈ G$, such that for any element $x ∈ G$, $x \circ e = e \circ x = x$.
The returned element is of a similar type to `x`.
"""
identity(M::Manifold, x) = identity(M, x, is_decorator_manifold(M))
identity(M::Manifold, x, ::Val{true}) = identity(M.manifold, x)
function identity(M::Manifold, x, ::Val{false})
    return error("identity not implemented on $(typeof(M)) for points $(typeof(x))")
end
function identity(G::AbstractGroupManifold, x)
    y = similar_result(G, identity, x)
    return identity!(G, y, x)
end

identity!(M::Manifold, y, x) = identity!(M, y, x, is_decorator_manifold(M))
identity!(M::Manifold, y, x, ::Val{true}) = identity!(M.manifold, y, x)
function identity!(M::Manifold, y, x, ::Val{false})
    return error("identity! not implemented on $(typeof(M)) for points $(typeof(y)) and $(typeof(x))")
end

isapprox(M::Manifold, x, e::Identity; kwargs...) = isapprox(M, e, x; kwargs...)
function isapprox(M::Manifold, e::Identity, x; kwargs...)
    is_decorator_group(M) === Val(true) && return isapprox(base_group(M), e, x; kwargs...)
    error("isapprox not implemented for manifold $(typeof(M)) and points $(typeof(e)) and $(typeof(x))")
end
function isapprox(M::Manifold, e::E, ::E; kwargs...) where {E<:Identity}
    is_decorator_group(M) === Val(true) && return isapprox(base_group(M), e, e; kwargs...)
    error("isapprox not implemented for manifold $(typeof(M)) and points $(typeof(e)) and $(typeof(e))")
end
function isapprox(G::GT, e::Identity{GT}, x; kwargs...) where {GT<:AbstractGroupManifold}
    return isapprox(G, identity(G, x), x; kwargs...)
end
function isapprox(
    ::GT,
    ::E,
    ::E;
    kwargs...,
) where {GT<:AbstractGroupManifold,E<:Identity{GT}}
    return true
end
function isapprox(G::GT, x, e::Identity{GT}; kwargs...) where {GT<:GroupManifold}
    return isapprox(G, e, x; kwargs...)
end
function isapprox(G::GT, e::Identity{GT}, x; kwargs...) where {GT<:GroupManifold}
    return isapprox(G, identity(G, x), x; kwargs...)
end
isapprox(::GT, ::E, ::E; kwargs...) where {GT<:GroupManifold,E<:Identity{GT}} = true

@doc doc"""
    compose(G::AbstractGroupManifold, x, y)

Compose elements $x,y ∈ G$ using the group operation $x \circ y$.
"""
compose(M::Manifold, x, y) = compose(M, x, y, is_decorator_manifold(M))
compose(M::Manifold, x, y, ::Val{true}) = compose(M.manifold, x, y)
function compose(M::Manifold, x, y, ::Val{false})
    return error("compose not implemented on $(typeof(M)) for elements $(typeof(x)) and $(typeof(y))")
end
function compose(G::AbstractGroupManifold, x, y)
    z = similar_result(G, compose, x, y)
    return compose!(G, z, x, y)
end

compose!(M::Manifold, z, x, y) = compose!(M, z, x, y, is_decorator_manifold(M))
compose!(M::Manifold, z, x, y, ::Val{true}) = compose!(M.manifold, z, x, y)
function compose!(M::Manifold, z, x, y, ::Val{false})
    return error("compose! not implemented on $(typeof(M)) for elements $(typeof(x)) and $(typeof(y))")
end

_action_order(x, y, conv::LeftAction) = (x, y)
_action_order(x, y, conv::RightAction) = (y, x)

@doc doc"""
    translate(G::AbstractGroupManifold, x, y[, conv::ActionDirection=LeftAction()])

For group elements $x,y ∈ G$, translate $y$ by $x$ with the specified convention, either
left $L_x$ or right $R_x$, defined as
```math
\begin{aligned}
L_x &: y ↦ x \circ y\\
R_x &: y ↦ y \circ x.
\end{aligned}
```
"""
translate(M::Manifold, x, y) = translate(M, x, y, LeftAction())
function translate(M::Manifold, x, y, conv::ActionDirection)
    return translate(M, x, y, conv, is_decorator_manifold(M))
end
function translate(M::Manifold, x, y, conv::ActionDirection, ::Val{true})
    return translate(M.manifold, x, y, conv)
end
function translate(M::Manifold, x, y, conv::ActionDirection, ::Val{false})
    return error("translate not implemented on $(typeof(M)) for elements $(typeof(x)) and $(typeof(y)) and direction $(typeof(conv))")
end
function translate(G::AbstractGroupManifold, x, y, conv::ActionDirection)
    return compose(G, _action_order(x, y, conv)...)
end

@doc doc"""
    translate!(G::AbstractGroupManifold, z, x, y[, conv::ActionDirection=LeftAction()])

For group elements $x,y ∈ G$, translate $y$ by $x$ with the specified convention, either
left $L_x$ or right $R_x$, defined as
```math
\begin{aligned}
L_x &: y ↦ x \circ y\\
R_x &: y ↦ y \circ x.
\end{aligned}
```
Result of the operation is saved in `z`.
"""
translate!(M::Manifold, z, x, y) = translate!(M, z, x, y, LeftAction())
function translate!(M::Manifold, z, x, y, conv::ActionDirection)
    return translate!(M, z, x, y, conv, is_decorator_manifold(M))
end
function translate!(M::Manifold, z, x, y, conv::ActionDirection, ::Val{true})
    return translate!(M.manifold, z, x, y, conv)
end
function translate!(M::Manifold, z, x, y, conv::ActionDirection, ::Val{false})
    return error("translate! not implemented on $(typeof(M)) for elements $(typeof(x)) and $(typeof(y)) and direction $(typeof(conv))")
end
function translate!(G::AbstractGroupManifold, z, x, y, conv::ActionDirection)
    return compose!(G, z, _action_order(x, y, conv)...)
end

@doc doc"""
    inverse_translate(G::AbstractGroupManifold, x, y, [conv::ActionDirection=Left()])

For group elements $x,y ∈ G$, inverse translate $y$ by $x$ with the specified convention,
either left $L_x^{-1}$ or right $R_x^{-1}$, defined as
```math
\begin{aligned}
L_x^{-1} &: y ↦ x^{-1} \circ y\\
R_x^{-1} &: y ↦ y \circ x^{-1}.
\end{aligned}
```
"""
inverse_translate(M::Manifold, x, y) = inverse_translate(M, x, y, LeftAction())
function inverse_translate(M::Manifold, x, y, conv::ActionDirection)
    return inverse_translate(M, x, y, conv, is_decorator_manifold(M))
end
function inverse_translate(M::Manifold, x, y, conv::ActionDirection, ::Val{true})
    return inverse_translate(M.manifold, x, y, conv)
end
function inverse_translate(M::Manifold, x, y, conv::ActionDirection, ::Val{false})
    return error("inverse_translate not implemented on $(typeof(M)) for elements $(typeof(x)) and $(typeof(y)) and direction $(typeof(conv))")
end
function inverse_translate(G::AbstractGroupManifold, x, y, conv::ActionDirection)
    return translate(G, inv(G, x), y, conv)
end

@doc doc"""
    inverse_translate!(G::AbstractGroupManifold, z, x, y, [conv::ActionDirection=Left()])

For group elements $x,y ∈ G$, inverse translate $y$ by $x$ with the specified convention,
either left $L_x^{-1}$ or right $R_x^{-1}$, defined as
```math
\begin{aligned}
L_x^{-1} &: y ↦ x^{-1} \circ y\\
R_x^{-1} &: y ↦ y \circ x^{-1}.
\end{aligned}
```
Result is saved in `z`.
"""
inverse_translate!(M::Manifold, z, x, y) = inverse_translate!(M, z, x, y, LeftAction())
function inverse_translate!(M::Manifold, z, x, y, conv::ActionDirection)
    return inverse_translate!(M, z, x, y, conv, is_decorator_manifold(M))
end
function inverse_translate!(M::Manifold, z, x, y, conv::ActionDirection, ::Val{true})
    return inverse_translate!(M.manifold, z, x, y, conv)
end
function inverse_translate!(M::Manifold, z, x, y, conv::ActionDirection, ::Val{false})
    return error("inverse_translate! not implemented on $(typeof(M)) for elements $(typeof(x)) and $(typeof(y)) and direction $(typeof(conv))")
end
function inverse_translate!(G::AbstractGroupManifold, z, x, y, conv::ActionDirection)
    return translate!(G, z, inv(G, x), y, conv)
end

@doc doc"""
    translate_diff(G::AbstractGroupManifold, x, y, v[, conv::ActionDirection=LeftAction()])

For group elements $x,y ∈ G$ and tangent vector $v ∈ T_y G$, compute the action of the
differential of the translation by $x$ on $v$, written as $(\mathrm{d}τ_x)_y (v)$, with the
specified left or right convention. The differential transports vectors:
```math
\begin{aligned}
(\mathrm{d}L_x)_y (v) &: T_y G → T_{x \circ y} G\\
(\mathrm{d}R_x)_y (v) &: T_y G → T_{y \circ x} G\\
\end{aligned}
```
"""
translate_diff(M::Manifold, x, y, v) = translate_diff(M, x, y, v, LeftAction())
function translate_diff(M::Manifold, x, y, v, conv::ActionDirection)
    return translate_diff(M, x, y, v, conv, is_decorator_manifold(M))
end
function translate_diff(M::Manifold, x, y, v, conv::ActionDirection, ::Val{true})
    return translate_diff(M.manifold, x, y, v, conv)
end
function translate_diff(M::Manifold, x, y, v, conv::ActionDirection, ::Val{false})
    return error("translate_diff not implemented on $(typeof(G)) for elements $(typeof(vout)), $(typeof(x)) and $(typeof(y)), vector $(typeof(v)), and direction $(typeof(conv))")
end
function translate_diff(G::AbstractGroupManifold, x, y, v, conv::ActionDirection)
    xy = translate(G, x, y, conv)
    vout = zero_tangent_vector(G, xy)
    translate_diff!(G, vout, x, y, v, conv)
    return vout
end

function translate_diff!(M::Manifold, vout, x, y, v)
    return translate_diff!(M, vout, x, y, v, LeftAction())
end
function translate_diff!(M::Manifold, vout, x, y, v, conv::ActionDirection)
    return translate_diff!(M, vout, x, y, v, conv, is_decorator_manifold(M))
end
function translate_diff!(M::Manifold, vout, x, y, v, conv::ActionDirection, ::Val{true})
    return translate_diff!(M.manifold, vout, x, y, v, conv)
end
function translate_diff!(M::Manifold, vout, x, y, v, conv::ActionDirection, ::Val{false})
    return error("translate_diff! not implemented on $(typeof(M)) for elements $(typeof(vout)), $(typeof(x)) and $(typeof(y)), vector $(typeof(v)), and direction $(typeof(conv))")
end

@doc doc"""
    inverse_translate_diff(G::AbstractGroupManifold, x, y, v[, conv::ActionDirection=Left()])

For group elements $x,y ∈ G$ and tangent vector $v ∈ T_y G$, compute the inverse of the
action of the differential of the translation by $x$ on $v$, written as
$((\mathrm{d}τ_x)_y)^{-1} (v) = (\mathrm{d}τ_{x^{-1}})_y (v)$, with the specified left or
right convention. The differential transports vectors:
```math
\begin{aligned}
((\mathrm{d}L_x)_y)^{-1} (v) &: T_y G → T_{x^{-1} \circ y} G\\
((\mathrm{d}R_x)_y)^{-1} (v) &: T_y G → T_{y \circ x^{-1}} G\\
\end{aligned}
```
"""
function inverse_translate_diff(M::Manifold, x, y, v)
    return inverse_translate_diff(M, x, y, v, LeftAction())
end
function inverse_translate_diff(M::Manifold, x, y, v, conv::ActionDirection)
    return inverse_translate_diff(M, x, y, v, conv, is_decorator_manifold(M))
end
function inverse_translate_diff(M::Manifold, x, y, v, conv::ActionDirection, ::Val{true})
    return inverse_translate_diff(M.manifold, x, y, v, conv)
end
function inverse_translate_diff(M::Manifold, x, y, v, conv::ActionDirection, ::Val{false})
    return error("inverse_translate_diff not implemented on $(typeof(M)) for elements $(typeof(vout)), $(typeof(x)) and $(typeof(y)), vector $(typeof(v)), and direction $(typeof(conv))")
end
function inverse_translate_diff(G::AbstractGroupManifold, x, y, v, conv::ActionDirection)
    return translate_diff(G, inv(G, x), y, v, conv)
end

function inverse_translate_diff!(M::Manifold, vout, x, y, v)
    return inverse_translate_diff!(M, vout, x, y, v, LeftAction())
end
function inverse_translate_diff!(M::Manifold, vout, x, y, v, conv::ActionDirection)
    return inverse_translate_diff!(M, vout, x, y, v, conv, is_decorator_manifold(M))
end
function inverse_translate_diff!(
    M::Manifold,
    vout,
    x,
    y,
    v,
    conv::ActionDirection,
    ::Val{true},
)
    return inverse_translate_diff!(M.manifold, vout, x, y, v, conv)
end
function inverse_translate_diff!(
    M::Manifold,
    vout,
    x,
    y,
    v,
    conv::ActionDirection,
    ::Val{false},
)
    return error("inverse_translate_diff! not implemented on $(typeof(M)) for elements $(typeof(vout)), $(typeof(x)) and $(typeof(y)), vector $(typeof(v)), and direction $(typeof(conv))")
end
function inverse_translate_diff!(
    G::AbstractGroupManifold,
    vout,
    x,
    y,
    v,
    conv::ActionDirection,
)
    return translate_diff!(G, vout, inv(G, x), y, v, conv)
end

@doc doc"""
    group_exp(G::AbstractGroupManifold, v)

Compute the group exponential of the Lie algebra element `v`.

Given an element $v ∈ 𝔤 = T_e G$, where $e$ is the [`identity`](@ref) element of the group
$G$, and $𝔤$ is its Lie algebra, the group exponential is the map

````math
\exp_e : 𝔤 → G,
````
such that for $t ∈ ℝ$, $γ(t) = \exp_e (t v)$ defines a one-parameter subgroup with the
following properties:

````math
\begin{aligned}
γ(t) &= γ(-t)^{-1}\\
γ(t + s) &= γ(t) \circ γ(s) = γ(s) \circ γ(t)\\
γ(0) &= e\\
\lim_{t → 0} \frac{d}{dt} γ(t) &= v.
\end{aligned}
````

In general, the group exponential is distinct from the Riemannian exponential [`exp`](@ref).

    group_exp(G::AbstractGroupManifold{AdditionOperation}, v)

Compute $y = v$.

    group_exp(G::AbstractGroupManifold{MultiplicationOperation}, v)

For number and `AbstractMatrix` types of `v`, compute the usual numeric/matrix exponential,

````math
\exp_e = \sum_{n=0}^\infty \frac{1}{n!} v^n.
````
"""
group_exp(M::Manifold, v) = group_exp(M, v, is_decorator_manifold(M))
group_exp(M::Manifold, v, ::Val{true}) = group_exp(M.manifold, v)
function group_exp(M::Manifold, v, ::Val{false})
    return error("group_exp not implemented on $(typeof(M)) for vector $(typeof(v)).")
end
function group_exp(G::AbstractGroupManifold, v)
    y = similar_result(G, group_exp, v)
    return group_exp!(G, y, v)
end

group_exp!(M::Manifold, y, v) = group_exp!(M, y, v, is_decorator_manifold(M))
group_exp!(M::Manifold, y, v, ::Val{true}) = group_exp!(M.manifold, y, v)
function group_exp!(M::Manifold, y, v, ::Val{false})
    return error("group_exp! not implemented on $(typeof(M)) for vector $(typeof(v)) and element $(typeof(y)).")
end

@doc doc"""
    group_log(G::AbstractGroupManifold, y)

Compute the group logarithm of the group element `y`.

Given an element $y ∈ G$, compute the right inverse of the exponential map
[`group_exp`](@ref), that is, the element $v ∈ 𝔤 = T_e G$, such that $y = \exp_e v$

    group_log(G::AbstractGroupManifold{AdditionOperation}, y)

Compute $v = y$.

    group_log(G::AbstractGroupManifold{MultiplicationOperation}, y)

For number and `AbstractMatrix` types of `y`, compute the usual numeric/matrix logarithm.
"""
group_log(M::Manifold, y) = group_log(M, y, is_decorator_manifold(M))
group_log(M::Manifold, y, ::Val{true}) = group_log(M.manifold, y)
function group_log(M::Manifold, y, ::Val{false})
    return error("group_log not implemented on $(typeof(M)) for element $(typeof(y)).")
end
function group_log(G::AbstractGroupManifold, y)
    v = similar_result(G, group_log, y)
    return group_log!(G, v, y)
end

group_log!(M::Manifold, v, y) = group_log!(M, v, y, is_decorator_manifold(M))
group_log!(M::Manifold, v, y, ::Val{true}) = group_log!(M.manifold, v, y)
function group_log!(M::Manifold, v, y, ::Val{false})
    return error("group_log! not implemented on $(typeof(M)) for element $(typeof(y)) and vector $(typeof(v)).")
end

############################
# Group-specific Retractions
############################

"""
    GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod

Retraction using the group exponential "translated" to any point on the manifold.

# Constructor

    GroupExponentialRetraction(conv::ActionDirection = LeftAction())
"""
struct GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod end

function GroupExponentialRetraction(conv::ActionDirection = LeftAction())
    return GroupExponentialRetraction{typeof(conv)}()
end

"""
    GroupLogarithmicInverseRetraction{D<:ActionDirection} <: AbstractInverseRetractionMethod

Retraction using the group logarithm "translated" to any point on the manifold.

# Constructor

    GroupLogarithmicInverseRetraction(conv::ActionDirection = LeftAction())
"""
struct GroupLogarithmicInverseRetraction{D<:ActionDirection} <:
       AbstractInverseRetractionMethod end

function GroupLogarithmicInverseRetraction(conv::ActionDirection = LeftAction())
    return GroupLogarithmicInverseRetraction{typeof(conv)}()
end

direction(::GroupExponentialRetraction{D}) where {D} = D()
direction(::GroupLogarithmicInverseRetraction{D}) where {D} = D()

@doc doc"""
    retract(
        G::AbstractGroupManifold,
        x,
        v,
        method::GroupExponentialRetraction{<:ActionDirection},
    )

Compute the retraction using the group exponential "translated" to any point on the
manifold. With a group translation $τ_x$ in a specified direction, the retraction is

````math
\operatorname{retr}_x = τ_x \circ \exp_e \circ (\mathrm{d}τ_x)_x^{-1}
````
"""
function retract(G::GroupManifold, x, v, method::GroupExponentialRetraction)
    conv = direction(method)
    vₑ = inverse_translate_diff(G, x, x, v, conv)
    yₑ = group_exp(G, vₑ)
    return translate(G, x, yₑ, conv)
end

function retract!(G::GroupManifold, y, x, v, method::GroupExponentialRetraction)
    return invoke(
        retract!,
        Tuple{Manifold,typeof(y),typeof(x),typeof(v),typeof(method)},
        G,
        y,
        x,
        v,
        method,
    )
end
function retract!(M::Manifold, y, x, v, method::GroupExponentialRetraction)
    conv = direction(method)
    vₑ = inverse_translate_diff(M, x, x, v, conv)
    yₑ = group_exp(M, vₑ)
    return translate!(M, y, x, yₑ, conv)
end

@doc doc"""
    inverse_retract(
        G::AbstractGroupManifold,
        x,
        v,
        method::GroupLogarithmicInverseRetraction{<:ActionDirection},
    )

Compute the inverse retraction using the group logarithm "translated" to any point on the
manifold. With a group translation $τ_x$ in a specified direction, the retraction is

````math
\operatorname{retr}_x^{-1} = (\mathrm{d}τ_x)_e \circ \log_e \circ τ_x^{-1}
````
"""
function inverse_retract(G::GroupManifold, x, y, method::GroupLogarithmicInverseRetraction)
    conv = direction(method)
    xinvy = inverse_translate(G, x, y, conv)
    vₑ = group_log(G, xinvy)
    return translate_diff(G, x, Identity(G), vₑ, conv)
end

function inverse_retract!(
    G::GroupManifold,
    v,
    x,
    y,
    method::GroupLogarithmicInverseRetraction,
)
    return invoke(
        inverse_retract!,
        Tuple{Manifold,typeof(v),typeof(x),typeof(y),typeof(method)},
        G,
        v,
        x,
        y,
        method,
    )
end
function inverse_retract!(M::Manifold, v, x, y, method::GroupLogarithmicInverseRetraction)
    conv = direction(method)
    xinvy = inverse_translate(M, x, y, conv)
    vₑ = group_log(M, xinvy)
    return translate_diff!(M, v, x, Identity(M), vₑ, conv)
end

#################################
# Overloads for AdditionOperation
#################################

"""
    AdditionOperation <: AbstractGroupOperation

Group operation that consists of simple addition.
"""
struct AdditionOperation <: AbstractGroupOperation end

const AdditionGroup = AbstractGroupManifold{AdditionOperation}

+(e::Identity{G}) where {G<:AdditionGroup} = e
+(::Identity{G}, x) where {G<:AdditionGroup} = x
+(x, ::Identity{G}) where {G<:AdditionGroup} = x
+(e::E, ::E) where {G<:AdditionGroup,E<:Identity{G}} = e

-(e::Identity{G}) where {G<:AdditionGroup} = e
-(::Identity{G}, x) where {G<:AdditionGroup} = -x
-(x, ::Identity{G}) where {G<:AdditionGroup} = x
-(e::E, ::E) where {G<:AdditionGroup,E<:Identity{G}} = e

*(e::Identity{G}, x) where {G<:AdditionGroup} = e
*(x, e::Identity{G}) where {G<:AdditionGroup} = e
*(e::E, ::E) where {G<:AdditionGroup,E<:Identity{G}} = e

zero(e::Identity{G}) where {G<:AdditionGroup} = e

identity(::AdditionGroup, x) = zero(x)

identity!(::AdditionGroup, y, x) = fill!(y, 0)

inv(::AdditionGroup, x) = -x

inv!(::AdditionGroup, y, x) = copyto!(y, -x)

compose(::AdditionGroup, x, y) = x + y

function compose!(::GT, z, x, y) where {GT<:AdditionGroup}
    x isa Identity{GT} && return copyto!(z, y)
    y isa Identity{GT} && return copyto!(z, x)
    z .= x .+ y
    return z
end

translate_diff(::AdditionGroup, x, y, v, ::ActionDirection) = v

translate_diff!(::AdditionGroup, vout, x, y, v, ::ActionDirection) = copyto!(vout, v)

inverse_translate_diff(::AdditionGroup, x, y, v, ::ActionDirection) = v

function inverse_translate_diff!(::AdditionGroup, vout, x, y, v, ::ActionDirection)
    return copyto!(vout, v)
end

group_exp(::AdditionGroup, v) = v

group_exp!(::AdditionGroup, y, v) = copyto!(y, v)

group_log(::AdditionGroup, y) = y

group_log!(::AdditionGroup, v, y) = copyto!(v, y)

#######################################
# Overloads for MultiplicationOperation
#######################################

"""
    MultiplicationOperation <: AbstractGroupOperation

Group operation that consists of multiplication.
"""
struct MultiplicationOperation <: AbstractGroupOperation end

const MultiplicationGroup = AbstractGroupManifold{MultiplicationOperation}

*(e::Identity{G}) where {G<:MultiplicationGroup} = e
*(::Identity{G}, x) where {G<:MultiplicationGroup} = x
*(x, ::Identity{G}) where {G<:MultiplicationGroup} = x
*(e::E, ::E) where {G<:MultiplicationGroup,E<:Identity{G}} = e

/(x, ::Identity{G}) where {G<:MultiplicationGroup} = x
/(::Identity{G}, x) where {G<:MultiplicationGroup} = inv(x)
/(e::E, ::E) where {G<:MultiplicationGroup,E<:Identity{G}} = e

\(x, ::Identity{G}) where {G<:MultiplicationGroup} = inv(x)
\(::Identity{G}, x) where {G<:MultiplicationGroup} = x
\(e::E, ::E) where {G<:MultiplicationGroup,E<:Identity{G}} = e

inv(e::Identity{G}) where {G<:MultiplicationGroup} = e

one(e::Identity{G}) where {G<:MultiplicationGroup} = e

transpose(e::Identity{G}) where {G<:MultiplicationGroup} = e

LinearAlgebra.det(::Identity{<:MultiplicationGroup}) = 1

LinearAlgebra.mul!(y, e::Identity{G}, x) where {G<:MultiplicationGroup} = copyto!(y, x)
LinearAlgebra.mul!(y, x, e::Identity{G}) where {G<:MultiplicationGroup} = copyto!(y, x)
function LinearAlgebra.mul!(y, e::E, ::E) where {G<:MultiplicationGroup,E<:Identity{G}}
    return identity!(e.group, y, e)
end

identity(::MultiplicationGroup, x) = one(x)

function identity!(G::GT, y, x) where {GT<:MultiplicationGroup}
    isa(x, Identity{GT}) || return copyto!(y, one(x))
    error("identity! not implemented on $(typeof(G)) for points $(typeof(y)) and $(typeof(x))")
end
identity!(::MultiplicationGroup, y::AbstractMatrix, x) = copyto!(y, I)

inv(::MultiplicationGroup, x) = inv(x)

inv!(G::MultiplicationGroup, y, x) = copyto!(y, inv(G, x))

compose(::MultiplicationGroup, x, y) = x * y

# TODO: z might alias with x or y, we might be able to optimize it if it doesn't.
compose!(::MultiplicationGroup, z, x, y) = copyto!(z, x * y)

inverse_translate(::MultiplicationGroup, x, y, ::LeftAction) = x \ y
inverse_translate(::MultiplicationGroup, x, y, ::RightAction) = y / x

function inverse_translate!(G::MultiplicationGroup, z, x, y, conv::ActionDirection)
    return copyto!(z, inverse_translate(G, x, y, conv))
end

function group_exp!(G::MultiplicationGroup, y, v)
    v isa Union{Number,AbstractMatrix} && return copyto!(y, exp(v))
    return error("group_exp! not implemented on $(typeof(G)) for vector $(typeof(v)) and element $(typeof(y)).")
end
