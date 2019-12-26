@doc doc"""
    AbstractGroupOperation

Abstract type for smooth binary operations $\mu$ on elements of a Lie group $G$.
```math
\mu \colon G \times G \to G.
```
An operation can be either defined for a specific [`AbstractGroupManifold`](@ref)
or in general, by defining for an operation `Op` the following methods:

    identity!(::AbstractGroupManifold{Op}, y, x)
    identity(::AbstractGroupManifold{Op}, x)
    inv!(::AbstractGroupManifold{Op}, y, x)
    inv(::AbstractGroupManifold{Op}, x)
    compose_left(::AbstractGroupManifold{Op}, x, y)
    compose_left!(::AbstractGroupManifold{Op}, z, x, y)

Note that a manifold is connected with an operation by wrapping it with
a decorator, [`AbstractGroupManifold`](@ref). In typical cases the concrete
wrapper [`GroupManifold`](@ref) can be used.
"""
abstract type AbstractGroupOperation end

@doc doc"""
    AbstractGroupManifold{<:AbstractGroupOperation} <: Manifold

Abstract type for a Lie group, a group that is also a smooth manifold with a
[smooth binary operation](@ref AbstractGroupOperation).
`AbstractGroupManifold`s must implement at least [`inv`](@ref),
[`identity`](@ref), [`compose_left`](@ref), and [`translate_diff`](@ref).
Group manifolds by default forward metric-related operations to the wrapped
manifold.
"""
abstract type AbstractGroupManifold{O<:AbstractGroupOperation} <: Manifold end

"""
    GroupManifold{M<:Manifold,O<:AbstractGroupOperation} <: AbstractGroupManifold{O}

Decorator for a smooth manifold that equips the manifold with a group
operation, thus making it a Lie group. See [`AbstractGroupManifold`](@ref) for
more details.

# Constructor

    GroupManifold(manifold, op)
"""
struct GroupManifold{M<:Manifold,O<:AbstractGroupOperation} <: AbstractGroupManifold{O}
    manifold::M
    op::O
end

is_decorator_manifold(::GroupManifold) = Val(true)

"""
    ActionDirection

Direction of action on a manifold, either [`LeftAction`](@ref) or
[`RightAction`](@ref).
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


@doc doc"""
    Identity(group::AbstractGroupManifold)

The identity element of the group `group`.
"""
struct Identity{G<:AbstractGroupManifold}
    group::G
end

(e::Identity)(x) = identity(e.group, x)

@doc doc"""
    inv(G::AbstractGroupManifold, x)

Inverse $x^{-1}$ of an element $x$, such that
$x \cdot x^{-1} = x^{-1} \cdot x = e$.
"""
function inv!(G::AbstractGroupManifold, y, x)
    error("inv not implemented on $(typeof(G)) for point $(typeof(x))")
end

function inv!(G::AbstractGroupManifold, y, e::Identity)
    indentity!(G, y)
    return y
end

function inv(G::AbstractGroupManifold, x)
    y = similar_result(G, inv, x)
    inv!(G, y, x)
    return y
end

inv(::AbstractGroupManifold, e::Identity) = e

function identity!(G::AbstractGroupManifold, y, x)
    error("identity! not implemented on $(typeof(G)) for points $(typeof(y)) and $(typeof(x))")
end

@doc doc"""
    identity(G::AbstractGroupManifold, x)

Identity element $e$, such that for any element $x$,
$x \cdot e = e \cdot x = x$. The returned element is of a similar type to `x`.
"""
function identity(G::AbstractGroupManifold, x)
    y = similar_result(G, inv, x)
    identity!(G, y, x)
    return y
end

# Adapted from `afoldl` in `operators.jl` in Julia base.
# expand recursively up to a point, then switch to a loop.
_group_afoldl(op, G, a) = a
_group_afoldl(op, G, a, b) = op(G,a,b)
_group_afoldl(op, G, a, b, c...) = _group_afoldl(op, G, op(G, a, b), c...)

function _group_afoldl(op,G,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,qs...)
    y = op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,a,b),c),d),e),f),g),h),i),j),k),l),m),n),o),p)
    return op(G, y, reduce((x, y)->op(G, x, y), qs))
end

"""
    compose_left(G::AbstractGroupManifold, xs...)

Compose elements of `G` using their left translation upon each other.
This composition is left-associative.
"""
compose_left(G::AbstractGroupManifold, xs...) = _group_afoldl(compose_left, G, xs...)
compose_left(G::AbstractGroupManifold, x) = x
function compose_left(G::AbstractGroupManifold, x, y)
    z = similar_result(G, compose_left, x, y)
    compose_left!(G, z, x, y)
    return z
end

"""
    compose_left!(G::AbstractGroupManifold, z, x, y)

Compose elements `x` and `y` of `G` using their left translation upon each other.
The result is saved in `z`.

`z` may alias with one or both of `x` and `y`.
"""
function compose_left!(G::AbstractGroupManifold, z, x, y)
    error("compose_left not implemented on $(typeof(G)) for elements $(typeof(x)) and $(typeof(y))")
end

@doc doc"""
    translate(G::AbstractGroupManifold, x, y[, conv::ActionDirection=LeftAction()])

For group elements $x,y \in G$, translate $y$ by $x$ with the specified
convention, either left $L_x$ or right $R_x$, defined as
```math
\begin{aligned}
L_x &\colon y \mapsto x \cdot y\\
R_x &\colon y \mapsto y \cdot x.
\end{aligned}
```
"""
translate(G::AbstractGroupManifold, x, y, conv::LeftAction) = compose_left(G, x, y)
translate(G::AbstractGroupManifold, x, y, conv::RightAction) = compose_left(G, y, x)
translate(G::AbstractGroupManifold, x, y) = translate(G, x, y, LeftAction())

@doc doc"""
    translate!(G::AbstractGroupManifold, z, x, y[, conv::ActionDirection=LeftAction()])

For group elements $x,y \in G$, translate $y$ by $x$ with the specified
convention, either left $L_x$ or right $R_x$, defined as
```math
\begin{aligned}
L_x &\colon y \mapsto x \cdot y\\
R_x &\colon y \mapsto y \cdot x.
\end{aligned}
```
Result of the operation is saved in `z`.
"""
translate!(G::AbstractGroupManifold, z, x, y, conv::LeftAction) = compose_left!(G, z, x, y)
translate!(G::AbstractGroupManifold, z, x, y, conv::RightAction) = compose_left!(G, z, y, x)
translate!(G::AbstractGroupManifold, z, x, y) = translate!(G, z, x, y, LeftAction())

@doc doc"""
    inverse_translate(G::AbstractGroupManifold, x, y, [conv::ActionDirection=Left()])

For group elements $x,y \in G$, inverse translate $y$ by $x$ with the specified
convention, either left $L_x^{-1}$ or right $R_x^{-1}$, defined as
```math
\begin{aligned}
L_x^{-1} &\colon y \mapsto x^{-1} \cdot y\\
R_x^{-1} &\colon y \mapsto y \cdot x^{-1}.
\end{aligned}
```
"""
function inverse_translate(G::AbstractGroupManifold,
                           x,
                           y,
                           conv::ActionDirection)
    return translate(G, inv(G, x), y, conv)
end

function inverse_translate(G::AbstractGroupManifold, x, y)
    return inverse_translate(G, x, y, LeftAction())
end

@doc doc"""
    inverse_translate!(G::AbstractGroupManifold, z, x, y, [conv::ActionDirection=Left()])

For group elements $x,y \in G$, inverse translate $y$ by $x$ with the specified
convention, either left $L_x^{-1}$ or right $R_x^{-1}$, defined as
```math
\begin{aligned}
L_x^{-1} &\colon y \mapsto x^{-1} \cdot y\\
R_x^{-1} &\colon y \mapsto y \cdot x^{-1}.
\end{aligned}
```
Result is saved in `z`.
"""
function inverse_translate!(G::AbstractGroupManifold,
                            z,
                            x,
                            y,
                            conv::ActionDirection)
    inv!(G, z, x)
    return translate!(G, z, z, y, conv)
end

function inverse_translate!(G::AbstractGroupManifold, z, x, y)
    return inverse_translate!(G, z, x, y, LeftAction())
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

function identity!(::AbstractGroupManifold{AdditionOperation}, y, x)
    fill!(y, 0)
    return y
end

identity(::AbstractGroupManifold{AdditionOperation}, x) = zero(x)

function inv!(::AbstractGroupManifold{AdditionOperation}, y, x)
    copyto!(y, -x)
    return y
end

inv(::AbstractGroupManifold{AdditionOperation}, x) = -x

compose_left(::AbstractGroupManifold{AdditionOperation}, x, y) = x + y
function compose_left!(::AbstractGroupManifold{AdditionOperation}, z, x, y)
    z .= x .+ y
    return z
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

# this is different from inv(G, e::Identity{G})
inv(e::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = e

function identity!(::AbstractGroupManifold{MultiplicationOperation}, y, x)
    copyto!(y, one(x))
    return y
end

identity(::AbstractGroupManifold{MultiplicationOperation}, x) = one(x)

function inv!(::AbstractGroupManifold{MultiplicationOperation}, y, x)
    copyto!(y, inv(x))
    return y
end

inv(::AbstractGroupManifold{MultiplicationOperation}, x) = inv(x)

compose_left(::AbstractGroupManifold{MultiplicationOperation}, x, y) = x * y
function compose_left!(::AbstractGroupManifold{MultiplicationOperation}, z, x, y)
    #TODO: z might alias with x or y, we might be able to optimize it if it doesn't.
    copyto!(z, x*y)
end
