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

# piping syntax for decoration
if VERSION ≥ v"1.3"
    (op::AbstractGroupOperation)(M::Manifold) = GroupManifold(M, op)
    (::Type{T})(M::Manifold) where {T<:AbstractGroupOperation} = GroupManifold(M, T())
end

@doc doc"""
    AbstractGroupManifold{<:AbstractGroupOperation} <: Manifold

Abstract type for a Lie group, a group that is also a smooth manifold with an
[`AbstractGroupOperation`](@ref), a smooth binary operation. `AbstractGroupManifold`s must
implement at least [`inv`](@ref), [`identity`](@ref), [`compose`](@ref), and
[`translate_diff`](@ref). Group manifolds by default forward metric-related operations to
the wrapped manifold.
"""
abstract type AbstractGroupManifold{O<:AbstractGroupOperation} <: Manifold end

"""
    GroupManifold{M<:Manifold,O<:AbstractGroupOperation} <: AbstractGroupManifold{O}

Decorator for a smooth manifold that equips the manifold with a group operation, thus making
it a Lie group. See [`AbstractGroupManifold`](@ref) for more details.

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

function check_manifold_point(G::GroupManifold, x; kwargs...)
    return check_manifold_point(G.manifold, x; kwargs...)
end

function check_tangent_vector(G::GroupManifold, x, v; kwargs...)
    return check_tangent_vector(G.manifold, x, v; kwargs...)
end

zero_tangent_vector(G::GroupManifold, x) = zero_tangent_vector(G.manifold, x)

zero_tangent_vector!(G::GroupManifold, y, x) = zero_tangent_vector!(G.manifold, y, x)

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
inv(::AG, e::Identity{AG}) where {AG<:AbstractGroupManifold} = e

identity!(M::Manifold, y, x) = identity!(M, y, x, is_decorator_manifold(M))
identity!(M::Manifold, y, x, ::Val{true}) = identity!(M.manifold, y, x)
function identity!(M::Manifold, y, x, ::Val{false})
    return error("identity! not implemented on $(typeof(M)) for points $(typeof(y)) and $(typeof(x))")
end

@doc doc"""
    identity(G::AbstractGroupManifold, x)

Identity element $e ∈ G$, such that for any element $x ∈ G$, $x \circ e = e \circ x = x$.
The returned element is of a similar type to `x`.
"""
function identity(G::AbstractGroupManifold, x)
    y = similar_result(G, identity, x)
    return identity!(G, y, x)
end
identity(::GT, e::Identity{GT}) where {GT<:AbstractGroupManifold} = e
identity(M::Manifold, x) = identity(M, x, is_decorator_manifold(M))
identity(M::Manifold, x, ::Val{true}) = identity(M.manifold, x)
function identity(M::Manifold, x, ::Val{false})
    return error("identity not implemented on $(typeof(M)) for points $(typeof(x))")
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
compose(::GT, ::Identity{GT}, y) where {GT<:AbstractGroupManifold} = y
compose(::GT, x, ::Identity{GT}) where {GT<:AbstractGroupManifold} = x
compose(::GT, x::Identity{GT}, ::Identity{GT}) where {GT<:AbstractGroupManifold} = x

compose!(M::Manifold, z, x, y) = compose!(M, z, x, y, is_decorator_manifold(M))
compose!(M::Manifold, z, x, y, ::Val{true}) = compose!(M.manifold, z, x, y)
function compose!(M::Manifold, z, x, y, ::Val{false})
    return error("compose! not implemented on $(typeof(M)) for elements $(typeof(x)) and $(typeof(y))")
end

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
translate(G::AbstractGroupManifold, x, y, conv::LeftAction) = compose(G, x, y)
translate(G::AbstractGroupManifold, x, y, conv::RightAction) = compose(G, y, x)

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
translate!(G::AbstractGroupManifold, z, x, y, conv::LeftAction) = compose!(G, z, x, y)
translate!(G::AbstractGroupManifold, z, x, y, conv::RightAction) = compose!(G, z, y, x)

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

"""
    AdditionOperation <: AbstractGroupOperation

Group operation that consists of simple addition.
"""
struct AdditionOperation <: AbstractGroupOperation end

+(e::Identity{G}) where {G<:AbstractGroupManifold{AdditionOperation}} = e
+(::Identity{G}, x) where {G<:AbstractGroupManifold{AdditionOperation}} = x
+(x, ::Identity{G}) where {G<:AbstractGroupManifold{AdditionOperation}} = x
+(e::E, ::E) where {G<:AbstractGroupManifold{AdditionOperation},E<:Identity{G}} = e

-(e::Identity{G}) where {G<:AbstractGroupManifold{AdditionOperation}} = e
-(::Identity{G}, x) where {G<:AbstractGroupManifold{AdditionOperation}} = -x
-(x, ::Identity{G}) where {G<:AbstractGroupManifold{AdditionOperation}} = x
-(e::E, ::E) where {G<:AbstractGroupManifold{AdditionOperation},E<:Identity{G}} = e

*(e::Identity{G}, x) where {G<:AbstractGroupManifold{AdditionOperation}} = e
*(x, e::Identity{G}) where {G<:AbstractGroupManifold{AdditionOperation}} = e
*(e::E, ::E) where {G<:AbstractGroupManifold{AdditionOperation},E<:Identity{G}} = e

function identity!(::AbstractGroupManifold{AdditionOperation}, y, x)
    fill!(y, 0)
    return y
end

identity(::AbstractGroupManifold{AdditionOperation}, x) = zero(x)
identity(::GT, e::Identity{GT}) where {GT<:AbstractGroupManifold{AdditionOperation}} = e

inv!(::AbstractGroupManifold{AdditionOperation}, y, x) = copyto!(y, -x)

zero(e::Identity{G}) where {G<:AbstractGroupManifold{AdditionOperation}} = e

inv(::AbstractGroupManifold{AdditionOperation}, x) = -x
inv(::AG, e::Identity{AG}) where {AG<:AbstractGroupManifold{AdditionOperation}} = e

compose(::AbstractGroupManifold{AdditionOperation}, x, y) = x + y
compose(::GT, x, ::Identity{GT}) where {GT<:AbstractGroupManifold{AdditionOperation}} = x
compose(::GT, ::Identity{GT}, y) where {GT<:AbstractGroupManifold{AdditionOperation}} = y
function compose(
    ::GT,
    x::Identity{GT},
    ::Identity{GT},
) where {GT<:AbstractGroupManifold{AdditionOperation}}
    return x
end

function compose!(::AbstractGroupManifold{AdditionOperation}, z, x, y)
    z .= x .+ y
    return z
end
function compose!(
    ::GT,
    z,
    x::Identity{GT},
    y,
) where {GT<:AbstractGroupManifold{AdditionOperation}}
    return copyto!(z, y)
end
function compose!(
    ::GT,
    z,
    x,
    y::Identity{GT},
) where {GT<:AbstractGroupManifold{AdditionOperation}}
    return copyto!(z, x)
end
function compose!(
    G::GT,
    z,
    e::Identity{GT},
    ::Identity{GT},
) where {GT<:AbstractGroupManifold{AdditionOperation}}
    return identity!(G, z, e)
end

function translate_diff(
    ::AbstractGroupManifold{AdditionOperation},
    x,
    y,
    v,
    ::ActionDirection,
)
    return v
end
function translate_diff!(
    ::AbstractGroupManifold{AdditionOperation},
    vout,
    x,
    y,
    v,
    ::ActionDirection,
)
    return copyto!(vout, v)
end

function inverse_translate_diff(
    ::AbstractGroupManifold{AdditionOperation},
    x,
    y,
    v,
    ::ActionDirection,
)
    return v
end
function inverse_translate_diff!(
    ::AbstractGroupManifold{AdditionOperation},
    vout,
    x,
    y,
    v,
    ::ActionDirection,
)
    return copyto!(vout, v)
end

"""
    MultiplicationOperation <: AbstractGroupOperation

Group operation that consists of multiplication.
"""
struct MultiplicationOperation <: AbstractGroupOperation end

*(e::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = e
*(::Identity{G}, x) where {G<:AbstractGroupManifold{MultiplicationOperation}} = x
*(x, ::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = x
*(e::E, ::E) where {G<:AbstractGroupManifold{MultiplicationOperation},E<:Identity{G}} = e

/(x, ::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = x
/(::Identity{G}, x) where {G<:AbstractGroupManifold{MultiplicationOperation}} = inv(x)
/(e::E, ::E) where {G<:AbstractGroupManifold{MultiplicationOperation},E<:Identity{G}} = e

\(x, ::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = inv(x)
\(::Identity{G}, x) where {G<:AbstractGroupManifold{MultiplicationOperation}} = x
\(e::E, ::E) where {G<:AbstractGroupManifold{MultiplicationOperation},E<:Identity{G}} = e

one(e::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = e

function LinearAlgebra.mul!(
    y,
    e::Identity{G},
    x,
) where {G<:AbstractGroupManifold{MultiplicationOperation}}
    return copyto!(y, x)
end
function LinearAlgebra.mul!(
    y,
    x,
    e::Identity{G},
) where {G<:AbstractGroupManifold{MultiplicationOperation}}
    return copyto!(y, x)
end
function LinearAlgebra.mul!(
    y,
    e::E,
    ::E,
) where {G<:AbstractGroupManifold{MultiplicationOperation},E<:Identity{G}}
    return identity!(e.group, y, e)
end

inv(e::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = e

identity!(::AbstractGroupManifold{MultiplicationOperation}, y, x) = copyto!(y, one(x))
function identity!(
    G::GT,
    y,
    x::Identity{GT},
) where {GT<:AbstractGroupManifold{MultiplicationOperation}}
    error("identity! not implemented on $(typeof(G)) for points $(typeof(y)) and $(typeof(x))")
end
function identity!(::AbstractGroupManifold{MultiplicationOperation}, y::AbstractMatrix, x)
    return copyto!(y, I)
end
function identity!(
    ::GT,
    y::AbstractMatrix,
    ::Identity{GT},
) where {GT<:AbstractGroupManifold{MultiplicationOperation}}
    return copyto!(y, I)
end

identity(::AbstractGroupManifold{MultiplicationOperation}, x) = one(x)
function identity(
    ::GT,
    e::Identity{GT},
) where {GT<:AbstractGroupManifold{MultiplicationOperation}}
    return e
end

inv!(::AbstractGroupManifold{MultiplicationOperation}, y, x) = copyto!(y, inv(x))

inv(::AbstractGroupManifold{MultiplicationOperation}, x) = inv(x)
inv(::AG, e::Identity{AG}) where {AG<:AbstractGroupManifold{MultiplicationOperation}} = e

compose(::AbstractGroupManifold{MultiplicationOperation}, x, y) = x * y
function compose(
    ::GT,
    x,
    ::Identity{GT},
) where {GT<:AbstractGroupManifold{MultiplicationOperation}}
    return x
end
function compose(
    ::GT,
    ::Identity{GT},
    y,
) where {GT<:AbstractGroupManifold{MultiplicationOperation}}
    return y
end
function compose(
    ::GT,
    x::Identity{GT},
    ::Identity{GT},
) where {GT<:AbstractGroupManifold{MultiplicationOperation}}
    return x
end

# TODO: z might alias with x or y, we might be able to optimize it if it doesn't.
compose!(::AbstractGroupManifold{MultiplicationOperation}, z, x, y) = copyto!(z, x * y)

function inverse_translate(
    ::AbstractGroupManifold{MultiplicationOperation},
    x,
    y,
    ::LeftAction,
)
    return x \ y
end
function inverse_translate(
    ::AbstractGroupManifold{MultiplicationOperation},
    x,
    y,
    ::RightAction,
)
    return y / x
end

function inverse_translate!(
    G::AbstractGroupManifold{MultiplicationOperation},
    z,
    x,
    y,
    conv::ActionDirection,
)
    return copyto!(z, inverse_translate(G, x, y, conv))
end
