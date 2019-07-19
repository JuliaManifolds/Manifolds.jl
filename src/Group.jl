"""
    AbstractGroupOperation

Abstract type for smooth binary operations on elements of a Lie group.
"""
abstract type AbstractGroupOperation end

@doc doc"""
    AbstractGroupManifold{<:AbstractGroupOperation} <: Manifold

Abstract type for a Lie group, a group that is also a smooth manifold with a
smooth binary operation.

`AbstractGroupManifold`s must implement at least [`inv`](@ref),
[`identity`](@ref), and [`left_action`](@ref).

Group manifolds by default assume a left-invariant canonical metric
$g_x(v,w) = g_e(x \cdot v, x \cdot w) = (x \cdot v)^T (x \cdot w)$, where
$g_x$ is the metric tensor at a point $x$, $g_e$ is the metric tensor at the
identity element $e$, and $x \cdot v$ is the left action of the $x$ on the
tangent vector $v$. This behavior can be changed by reimplementing
[`inner`](@ref), [`exp!`](@ref), and [`log!`](@ref).
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
struct GroupManifold{M<:Manifold,O} <: AbstractGroupManifold{O}
    manifold::M
    op::O
end

@traitimpl IsDecoratorManifold{GroupManifold}

struct LeftInvariantCanonicalMetric <: RiemannianMetric end
struct RightInvariantCanonicalMetric <: RiemannianMetric end

@traitimpl HasMetric{GroupManifold,LeftInvariantCanonicalMetric}

"""
    IsMatrixGroup

A `Trait` to indicate that a group's default representation is a matrix
representation. For groups with a `MultiplicationOperation`, this indicates
that the group is a subgroup of the General Linear Group.
"""
@traitdef IsMatrixGroup{G}

@doc doc"""
    Identity

The identity element of any group.
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
function inv(G::AbstractGroupManifold, x)
    error("inv not implemented on $(typeof(G)) for point $(typeof(x))")
end

inv(::AbstractGroupManifold, e::Identity) = e

@doc doc"""
    identity(G::AbstractGroupManifold, x)

Identity element $e$, such that for any element $x$,
$x \cdot e = e \cdot x = x$. The returned element is of a similar type to `x`.
"""
function identity(G::AbstractGroupManifold, x)
    error("identity not implemented on $(typeof(G)) for point $(typeof(x))")
end

@doc doc"""
    left_action(G::AbstractGroupManifold, x, p)

For an element $p$ of some set, compute the left action of group element $x$
on $p$, i.e. $x \cdot p$.
"""
function left_action(G::AbstractGroupManifold, x, p)
    error("left_action not implemented on $(typeof(G)) for point $(typeof(x)) and object $(typeof(p))")
end

@doc doc"""
    right_action(G::AbstractGroupManifold, x, p)

For an element $p$ of some set, compute the right action of group element $x$
on $p$, i.e. $p \cdot x$. The default right action is constructed from the left
action, i.e. $p \cdot x = x^{-1} \cdot p$.
"""
function right_action(G::AbstractGroupManifold, x, p)
    return left_action(G, inv(G, x), p)
end

# Adapted from `afoldl` in `operators.jl` in Julia base.
# expand recursively up to a point, then switch to a loop.
group_afoldl(op, G, a) = a
group_afoldl(op, G, a, b) = op(G,a,b)
group_afoldl(op, G, a, b, c...) = group_afoldl(op, G, op(G, a, b), c...)

function group_afoldl(op,G,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,qs...)
    y = op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,op(G,a,b),c),d),e),f),g),h),i),j),k),l),m),n),o),p)
    return op(G, y, reduce((x, y)->op(G, x, y), qs))
end

"""
    compose(G::AbstractGroupManifold, xs...)

Compose elements of `G` using their left translation upon each other.
"""
compose(G::AbstractGroupManifold, xs...) = group_afoldl(compose, G, xs...)
compose(G::AbstractGroupManifold, x) = x
compose(G::AbstractGroupManifold, x, y) = left_action(G, x, y)

*(G::AbstractGroupManifold, x, y) = compose(G, x, y)

/(G::AbstractGroupManifold, x, y) = compose(G, x, inv(G, y))

\(G::AbstractGroupManifold, x, y) = compose(G, inv(G, x), y)

@doc doc"""
    conjugate(G::AbstractGroupManifold, x, y)

Compute the conjugate of the element $y$ by element $x$, i.e.
$x \cdot y \cdot x^{-1}$
"""
function conjugate(G::AbstractGroupManifold, x, y)
    return compose(G, x, y, inv(G, x))
end

function adjoint_representation(G::AbstractGroupManifold, x, v)
    return left_action(G, left_action(G, x, v), inv(G, x))
end

function adjoint_representation_derivative(G::AbstractGroupManifold, v, w)
    return lie_bracket(G, v, w)
end

function lie_bracket(G::AbstractGroupManifold, v, w)
    error("lie_bracket not implemented on $(typeof(G)) for vectors $(typeof(v)) and $(typeof(w))")
end

function inner(G::AbstractGroupManifold, x, v, w)
    return dot(left_action(G, x, v), left_action(G, x, w))
end

"""
    exp!(G::AbstractGroupManifold, y, v)

Exponential map of tangent vector `v` at the identity element of group `G`.
Result is saved to `y`.
"""
function exp!(G::AbstractGroupManifold, y, x::Identity, v)
    error("Exponential map not implemented on $(typeof(G)) for identity $(typeof(x)), point $(typeof(y)), and tangent vector $(typeof(v))")
end

function exp!(G::AbstractGroupManifold, y, x, v)
    exp!(G, y, Identity(G), \(G, x, v))
    y .= compose(G, x, y)
    return y
end

function log!(G::AbstractGroupManifold, v, x::Identity, y)
    error("Logarithmic map not implemented on $(typeof(G)) for identity $(typeof(x)), point $(typeof(y)), and vector $(typeof(v))")
end

function log!(G::AbstractGroupManifold, v, x, y)
    log!(G, v, Identity(G), \(G, x, y))
    v .= left_action(G, x, v)
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

identity(::AbstractGroupManifold{AdditionOperation}, x) = zero(x)

inv(::AbstractGroupManifold{AdditionOperation}, x) = -x

left_action(::AbstractGroupManifold{AdditionOperation}, x, p) = x + p

@doc doc"""
    right_action(::AbstractGroupManifold{AdditionOperation}, x, p)

For an element $p$ of some set, compute the right action of group element $x$
on $p$, i.e. $p \cdot x$. For an `AdditionOperation`, the right and left
actions are the same.
"""
right_action(::AbstractGroupManifold{AdditionOperation}, x, p) = p + x

lie_bracket(::AbstractGroupManifold{AdditionOperation}, v, w) = zero(v)

function exp!(::GT,
              y,
              ::Identity{GT},
              v) where {GT<:AbstractGroupManifold{AdditionOperation}}
    y .= v
    return y
end

function log!(::GT,
              v,
              ::Identity{GT},
              y) where {GT<:AbstractGroupManifold{AdditionOperation}}
    v .= y
    return v
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

inv(e::Identity{G}) where {G<:AbstractGroupManifold{AdditionOperation}} = e

identity(::AbstractGroupManifold{MultiplicationOperation}, x) = one(x)

inv(::AbstractGroupManifold{MultiplicationOperation}, x) = inv(x)
inv(::AbstractGroupManifold{MultiplicationOperation}, x::Identity) = x

left_action(::AbstractGroupManifold{MultiplicationOperation}, x, p) = x * p

/(::AbstractGroupManifold{MultiplicationOperation}, x, y) = x / y

\(::AbstractGroupManifold{MultiplicationOperation}, x, y) = x \ y

@traitfn function lie_bracket(G::GT, v, w) where {O<:MultiplicationOperation,
                                                  GT<:AbstractGroupManifold{O};
                                                  !IsMatrixGroup{GT}}
    error("lie_bracket not implemented on $(GT) for vectors $(typeof(v)) and $(typeof(w))")
end

@traitfn function lie_bracket(G::GT, v, w) where {O<:MultiplicationOperation,
                                                  GT<:AbstractGroupManifold{O};
                                                  IsMatrixGroup{GT}}
    return v * w - w * v
end

@traitfn function exp!(G::GT,
                       y,
                       ::Identity{GT},
                       v) where {O<:MultiplicationOperation,
                                 GT<:AbstractGroupManifold{O};
                                 !IsMatrixGroup{GT}}
    error("Exponential map not implemented on $(typeof(G)) for identity $(typeof(x)), point $(typeof(y)), and tangent vector $(typeof(v))")
end

@traitfn function exp!(G::GT,
                       y,
                       ::Identity{GT},
                       v) where {O<:MultiplicationOperation,
                                 GT<:AbstractGroupManifold{O};
                                 IsMatrixGroup{GT}}
    y .= exp(v)
    return y
end

_log(x) = log(x)
_log(x::StaticMatrix) = log(Matrix(x))

@traitfn function log!(G::GT,
                       v,
                       ::Identity{GT},
                       y) where {O<:MultiplicationOperation,
                                 GT<:AbstractGroupManifold{O};
                                 !IsMatrixGroup{GT}}
    error("Logarithmic map not implemented on $(typeof(G)) for identity $(typeof(x)), point $(typeof(y)), and vector $(typeof(v))")
end

@traitfn function log!(G::GT,
                       v,
                       ::Identity{GT},
                       y) where {O<:MultiplicationOperation,
                                 GT<:AbstractGroupManifold{O};
                                 IsMatrixGroup{GT}}
    w = _log(y)
    v .= isreal(v) ? real(w) : w
    return v
end
