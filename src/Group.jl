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
[`identity`](@ref), [`compose`](@ref), and [`translate_diff`](@ref).

Group manifolds by default assume a left-invariant canonical metric.
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

abstract type ActionDirection end

struct Left <: ActionDirection end
struct Right <: ActionDirection end

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
compose(G::AbstractGroupManifold, x, y) = error("compose not implemented on $(typeof(G)) for elements $(typeof(x)) and $(typeof(y))")

@doc doc"""
    translate(G::AbstractGroupManifold, x, y, [conv::ActionDirection=Left()])

For group elements $x,y \in G$, translate $y$ by $x$ with the specified
convention, either left $L_x$ or right $R_x$, defined as

```math
\begin{aligned}
L_x &\colon y \mapsto x \cdot y\\
R_x &\colon y \mapsto y \cdot x.
\end{aligned}
```
"""
translate(G::AbstractGroupManifold, x, y, conv::Left) = compose(G, x, y)
translate(G::AbstractGroupManifold, x, y, conv::Right) = compose(G, y, x)
translate(G::AbstractGroupManifold, x, y) = translate(G, x, y, Left())

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
    return inverse_translate(G, x, y, Left())
end

@doc doc"""
    translate_diff(G::AbstractGroupManifold, x, y, vy, [conv::ActionDirection=Left()])

For group elements $x,y \in G$ and tangent vector $v_y \in T_y G$, compute the
action of the differential of the translation by $x$ on $v_y$, written as
$(d\tau_x)_y (v_y)$, with the specified left or right convention. The
differential transports vectors:

```math
\begin{aligned}
(dL_x)_y (v_y) &\colon T_y G \to T_{x \cdot y} G\\
(dR_x)_y (v_y) &\colon T_y G \to T_{y \cdot x} G\\
\end{aligned}
```
"""
function translate_diff(G::AbstractGroupManifold,
                        x,
                        y,
                        vy,
                        conv::ActionDirection)
    return error("translate_diff not implemented on $(typeof(G)) for elements $(typeof(x)) and $(typeof(y)), vector $(typeof(vy)), and direction $(typeof(conv))")
end

translate_diff(G, x, y, vy) = translate_diff(G, x, y, vy, Left())

@doc doc"""
    inverse_translate_diff(G::AbstractGroupManifold, x, y, vy, [conv::ActionDirection=Left()])

For group elements $x,y \in G$ and tangent vector $v_y \in T_y G$, compute the
inverse of the action of the differential of the translation by $x$ on $v_y$,
written as $((d\tau_x)_y)^{-1} (v_y) = (d\tau_{x^{-1}})_y (v_y)$, with the
specified left or right convention. The differential transports vectors:

```math
\begin{aligned}
((dL_x)_y)^{-1} (v_y) &\colon T_y G \to T_{x^{-1} \cdot y} G\\
((dR_x)_y)^{-1} (v_y) &\colon T_y G \to T_{y \cdot x^{-1}} G\\
\end{aligned}
```
"""
function inverse_translate_diff(G::AbstractGroupManifold,
                                x,
                                y,
                                vy,
                                conv::ActionDirection)
    return translate_diff(G, inv(G, x), y, vy, conv)
end

inverse_translate_diff(G, x, y, vy) = inverse_translate_diff(G, x, y, vy, Left())

function vector_transport!(G::AbstractGroupManifold, vy, x, vx, y)
    yxinv = inverse_translate(G, x, y, Right())
    vy .= translate_diff(G, yxinv, x, vx, Left())
    return vy
end

@doc doc"""
    conj(G::AbstractGroupManifold, x, y)

For group elements $x,y \in G$, compute the group homomorphism $\phi_x$

$\phi_x \colon y \mapsto x \cdot y \cdot x^{-1},$

called left conjugation of $y$ by $x$.
"""
function conj(G::AbstractGroupManifold, x, y)
    return translate(G, x, inverse_translate(G, x, y, Right()), Left())
end

@doc doc"""
    conj_diff(G::AbstractGroupManifold, x, y, vy)

For group elements $x,y \in G$ and vector $v_y \in T_y G$, compute the
action of the differential of the conjugation [`conj`](@ref) $\phi_x$ at $y$ on
$v_y$, written as $(d\phi_x)_y (v_y)$.
"""
function conj_diff(G::AbstractGroupManifold, x, y, vy)
    yxinv = inverse_translate(G, x, y, Right())
    vyxinv = inverse_translate_diff(G, x, y, vy, Right())
    return translate_diff(G, x, vyxinv, vyxinv, Left())
end

@doc doc"""
    adjoint(G::AbstractGroupManifold, x, ve)

For group element $x \in G$ and vector $v_e \in \mathfrak{g} = T_e G$, compute
the action of the adjoint representation $\operatorname{Ad}_x$ of $G$ at $x$ on
$v_e$, defined as a special case of [`conj_diff`](@ref):

$\operatorname{Ad}_x = (d\phi_x)_e \colon \mathfrak{g} \to \mathfrak{g}$
"""
adjoint(G::AbstractGroupManifold, x, ve) = conj_diff(G, x, Identity(G), ve)

@doc doc"""
    adjoint_diff(G::AbstractGroupManifold, ve, we)

For Lie algebra vectors $v, w \in \mathfrak{g} = T_e G$, compute the
the action of the adjoint representation $\operatorname{ad}_v$ of
$\mathfrak{g}$ at $v$ on $w$. $\operatorname{ad}_v$ is defined as the derivative
of the [`adjoint`](@ref) representation on the group:

$\operatorname{ad}_v (w) = d(\operatorname{Ad})_e (v) (w)$
"""
adjoint_diff(G, ve, we) = lie_bracket(G, ve, we)

@doc doc"""
    lie_bracket(G, ve, we)

Compute the Lie bracket of the vectors $v,w \in T_e G = \mathfrak{g}$.
"""
function lie_bracket(G, ve, we)
    error("lie_bracket not implemented on $(typeof(G)) for vectors $(typeof(ve)) and $(typeof(we))")
end

function inner(G::AbstractGroupManifold, e::Identity, ve, we)
    error("inner not implemented on $(typeof(G)) for identity $(typeof(e)) and vectors $(typeof(ve)) and $(typeof(we))")
end

@doc doc"""
    inner(G::AbstractGroupManifold, x, vx, wx)

Compute the inner product of vectors $v_x, w_x \in T_x G$, assuming a left-
invariant metric, i.e.

$g(v_x, w_x)_x = g((dL_{x^{-1}})_x v_x, (dL_{x^{-1}})_x w_x)_e = g(v_e, w_e)_e$
"""
function inner(G::AbstractGroupManifold, x, vx, wx)
    return inner(G,
                 Identity(G),
                 inverse_translate_diff(G, x, x, vx, Left()),
                 inverse_translate_diff(G, x, x, wx, Left()),
                )
end

"""
    exp!(G::AbstractGroupManifold, h, e::Identity, ve)

Exponential map of tangent vector `ve` at the identity element `e` of group `G`.
Result is saved to `y`.
"""
function exp!(G::AbstractGroupManifold, y, e::Identity, ve)
    error("Exponential map not implemented on $(typeof(G)) for identity $(typeof(e)), point $(typeof(y)), and tangent vector $(typeof(ve))")
end

function exp!(G::AbstractGroupManifold, y, x, vx)
    ve = inverse_translate_diff(G, x, x, vx, Left())
    exp!(G, y, Identity(G), ve)
    copyto!(y, translate(G, x, y, Left()))
    return y
end

function log!(G::AbstractGroupManifold, ve, e::Identity, y)
    error("Logarithmic map not implemented on $(typeof(G)) for identity $(typeof(e)), point $(typeof(y)), and vector $(typeof(ve))")
end

function log!(G::AbstractGroupManifold, vx, x, y)
    e = Identity(G)
    xinvy = inverse_translate(G, x, y, Left())
    log!(G, vx, e, xinvy)
    vx .= translate_diff(G, x, e, vx, Left())
    return vx
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

compose(::AbstractGroupManifold{AdditionOperation}, x, y) = x + y

translate_diff(::AbstractGroupManifold{AdditionOperation}, x, y, vy, ::Left) = vy
translate_diff(::AbstractGroupManifold{AdditionOperation}, x, y, vy, ::Right) = vy

lie_bracket(::AbstractGroupManifold{AdditionOperation}, v, w) = zero(v)

function exp!(::GT,
              y,
              ::Identity{GT},
              ve) where {GT<:AbstractGroupManifold{AdditionOperation}}
    y .= ve
    return y
end

function log!(::GT,
              ve,
              ::Identity{GT},
              y) where {GT<:AbstractGroupManifold{AdditionOperation}}
    ve .= y
    return ve
end

"""
    MultiplicationOperation <: AbstractGroupOperation

Group operation that consists of multiplication.
"""
struct MultiplicationOperation <: AbstractGroupOperation end

*(e::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = e
*(::Identity{G}, x) where {G<:AbstractGroupManifold{MultiplicationOperation}} = x
*(x, ::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = x
*(e::E, ::E) where {G<:AbstractGroupManifold{MultiplicationOperation},E<:Identity{G}} = x

/(x, ::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = x
/(::Identity{G}, x) where {G<:AbstractGroupManifold{MultiplicationOperation}} = inv(x)
/(e::E, ::E) where {G<:AbstractGroupManifold{MultiplicationOperation},E<:Identity{G}} = e

\(x, ::Identity{G}) where {G<:AbstractGroupManifold{MultiplicationOperation}} = inv(x)
\(::Identity{G}, g) where {G<:AbstractGroupManifold{MultiplicationOperation}} = x
\(e::E, ::E) where {G<:AbstractGroupManifold{MultiplicationOperation},E<:Identity{G}} = e

inv(e::Identity{G}) where {G<:AbstractGroupManifold{AdditionOperation}} = e

identity(::AbstractGroupManifold{MultiplicationOperation}, x) = one(x)

inv(::AbstractGroupManifold{MultiplicationOperation}, x) = inv(x)

function translate_diff(::AbstractGroupManifold{MultiplicationOperation},
                        x,
                        y,
                        vy,
                        ::Left)
    return x * vy
end

function translate_diff(::AbstractGroupManifold{MultiplicationOperation},
                        x,
                        y,
                        vy,
                        ::Right)
    return vy * x
end

@traitfn function lie_bracket(G::GT, ve, we) where {O<:MultiplicationOperation,
                                                    GT<:AbstractGroupManifold{O};
                                                    !IsMatrixGroup{GT}}
    error("lie_bracket not implemented on $(GT) for vectors $(typeof(ve)) and $(typeof(we))")
end

@traitfn function lie_bracket(G::GT, ve, we) where {O<:MultiplicationOperation,
                                                   GT<:AbstractGroupManifold{O};
                                                   IsMatrixGroup{GT}}
    return ve * we - we * ve
end

@traitfn function exp!(G::GT,
                       y,
                       ::Identity{GT},
                       ve) where {O<:MultiplicationOperation,
                                  GT<:AbstractGroupManifold{O};
                                  !IsMatrixGroup{GT}}
    error("Exponential map not implemented on $(typeof(G)) for identity $(typeof(x)), point $(typeof(y)), and tangent vector $(typeof(ve))")
end

@traitfn function exp!(G::GT,
                       y,
                       ::Identity{GT},
                       ve) where {O<:MultiplicationOperation,
                                  GT<:AbstractGroupManifold{O};
                                  IsMatrixGroup{GT}}
    y .= exp(ve)
    return y
end

@traitfn function log!(G::GT,
                       ve,
                       ::Identity{GT},
                       y) where {O<:MultiplicationOperation,
                                 GT<:AbstractGroupManifold{O};
                                 !IsMatrixGroup{GT}}
    error("Logarithmic map not implemented on $(typeof(G)) for identity $(typeof(x)), point $(typeof(y)), and vector $(typeof(ve))")
end

@traitfn function log!(G::GT,
                       ve,
                       ::Identity{GT},
                       y) where {O<:MultiplicationOperation,
                                 GT<:AbstractGroupManifold{O};
                                 IsMatrixGroup{GT}}
    w = log_safe(y)
    ve .= isreal(ve) ? real(w) : w
    return ve
end
