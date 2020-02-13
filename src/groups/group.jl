@doc raw"""
    AbstractGroupOperation

Abstract type for smooth binary operations $∘$ on elements of a Lie group $\mathcal{G}$:
```math
∘ : \mathcal{G} × \mathcal{G} → \mathcal{G}
```
An operation can be either defined for a specific [`AbstractGroupManifold`](@ref)
or in general, by defining for an operation `Op` the following methods:

    identity!(::AbstractGroupManifold{Op}, q, q)
    identity(::AbstractGroupManifold{Op}, p)
    inv!(::AbstractGroupManifold{Op}, q, p)
    inv(::AbstractGroupManifold{Op}, p)
    compose(::AbstractGroupManifold{Op}, p, q)
    compose!(::AbstractGroupManifold{Op}, x, p, q)

Note that a manifold is connected with an operation by wrapping it with a decorator,
[`AbstractGroupManifold`](@ref). In typical cases the concrete wrapper
[`GroupManifold`](@ref) can be used.
"""
abstract type AbstractGroupOperation end

@doc raw"""
    AbstractGroupManifold{<:AbstractGroupOperation} <: Manifold

Abstract type for a Lie group, a group that is also a smooth manifold with an
[`AbstractGroupOperation`](@ref), a smooth binary operation. `AbstractGroupManifold`s must
implement at least [`inv`](@ref), [`identity`](@ref), [`compose`](@ref), and
[`translate_diff`](@ref).
"""
abstract type AbstractGroupManifold{O<:AbstractGroupOperation} <: AbstractDecoratorManifold end

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

"""
    base_group(M::Manifold) -> AbstractGroupManifold

Un-decorate `M` until an `AbstractGroupManifold` is encountered.
Return an error if the [`base_manifold`](@ref) is reached without encountering a group.
"""
base_group(M::DT) where {DT<:AbstractDecoratorManifold} = base_group(M.manifold)
function base_group(M::Manifold)
    error("base_group: no base group found.")
end
base_group(G::AbstractGroupManifold) = G

decorator_group_dispatch(M::Manifold) = Val(false)
decorator_group_dispatch(M::AbstractDecoratorManifold) = decorator_group_dispatch(M.manifold)
decorator_group_dispatch(M::AbstractGroupManifold) = Val(true)

function is_group_decorator(M::Manifold)
    return _extract_val(decorator_group_dispatch(M))
end

default_decorator_dispach(M::AbstractGroupManifold) = Val(false)

# piping syntax for decoration
if VERSION ≥ v"1.3"
    (op::AbstractGroupOperation)(M::Manifold) = GroupManifold(M, op)
    (::Type{T})(M::Manifold) where {T<:AbstractGroupOperation} = GroupManifold(M, T())
end

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

@doc raw"""
    Identity(G::AbstractGroupManifold)

The group identity element $e ∈ \mathcal{G}$.
"""
struct Identity{G<:AbstractGroupManifold}
    group::G
end

Identity(M::DT) where {DT <: AbstractDecoratorManifold}= Identity(M.manifold)
Identity(M::Manifold) = error("Identity not implemented for manifold $(M)")

show(io::IO, e::Identity) = print(io, "Identity($(e.group))")

(e::Identity)(p) = identity(e.group, p)

# To ensure allocate_result_type works
number_eltype(e::Identity) = Bool

copyto!(e::TE, ::TE) where {TE<:Identity} = e
copyto!(p, ::TE) where {TE<:Identity} = identity!(e.group, p, e)
copyto!(p::AbstractArray, e::TE) where {TE<:Identity} = identity!(e.group, p, e)

isapprox(p, e::Identity; kwargs...) = isapprox(e::Identity, p; kwargs...)
isapprox(e::Identity, p; kwargs...) = isapprox(e.group, e, p; kwargs...)
isapprox(e::E, ::E; kwargs...) where {E<:Identity} = true

function allocate_result(M::Manifold, ::typeof(hat), e::Identity, Xⁱ)
    is_group_decorator(M) && return allocate_result(base_group(M), hat, e, Xⁱ)
    error("allocate_result not implemented for manifold $(M), function hat, point $(e), and vector $(Xⁱ).")
end
function allocate_result(
    G::GT,
    ::typeof(hat),
    ::Identity{GT},
    Xⁱ,
) where {GT<:AbstractGroupManifold}
    B = VectorBundleFibers(TangentSpace, G)
    return allocate(Xⁱ, Size(representation_size(B)))
end
function allocate_result(M::Manifold, ::typeof(vee), e::Identity, X)
    is_group_decorator(M) && return allocate_result(base_group(M), vee, e, X)
    error("allocate_result not implemented for manifold $(M), function vee, point $(e), and vector $(X).")
end
function allocate_result(
    G::GT,
    ::typeof(vee),
    ::Identity{GT},
    X,
) where {GT<:AbstractGroupManifold}
    return allocate(X, Size(manifold_dimension(G)))
end

function check_manifold_point(M::DT, e::Identity; kwargs...) where {DT<:AbstractDecoratorManifold}
    check_manifold_point(M.manifold, e; kwargs...)
end
function check_manifold_point(M::Manifold, e::Identity; kwargs...)
    return DomainError(e, "The identity element $(e) does not belong to $(M).")
end
function check_manifold_point(G::GroupManifold, e::Identity; kwargs...)
    e === Identity(G) && return nothing
    return DomainError(e, "The identity element $(e) does not belong to $(G).")
end

##########################
# Group-specific functions
##########################

@doc raw"""
    inv(G::AbstractGroupManifold, p)

Inverse $p^{-1} ∈ \mathcal{G}$ of an element $p ∈ \mathcal{G}$, such that
$p \circ p^{-1} = p^{-1} \circ p = e ∈ \mathcal{G}$, where $e$ is the [`identity`](@ref)
element of $\mathcal{G}$.
"""
inv(M::DT, p) where {DT <: AbstractDecoratorManifold} = inv(M.manifold, p)
function inv(M::Manifold, p)
    error("inv not implemented on $(typeof(M)) for points $(typeof(p))")
end
function inv(G::AbstractGroupManifold, p)
    q = allocate_result(G, inv, p)
    return inv!(G, q, p)
end

inv!(M::DT, q, p) where {DT <: AbstractDecoratorManifold} = inv!(M.manifold, q, p)
function inv!(M::Manifold, q, p)
    return error("inv! not implemented on $(typeof(M)) for points $(typeof(p))")
end

@doc raw"""
    identity(G::AbstractGroupManifold, p)

Identity element $e ∈ \mathcal{G}$, such that for any element $p ∈ \mathcal{G}$,
$p \circ e = e \circ p = p$.
The returned element is of a similar type to `p`.
"""
identity(M::DT, p) where {DT <: AbstractDecoratorManifold} = identity(M.manifold, p)
function identity(M::Manifold, p, ::Val{false})
    return error("identity not implemented on $(typeof(M)) for points $(typeof(p))")
end
function identity(G::AbstractGroupManifold, p)
    y = allocate_result(G, identity, p)
    return identity!(G, y, p)
end

identity!(M::AbstractDecoratorManifold, q, p) = identity!(M.manifold, q, p)
function identity!(M::Manifold, q, p)
    return error("identity! not implemented on $(typeof(M)) for points $(typeof(q)) and $(typeof(p))")
end

isapprox(M::Manifold, p, e::Identity; kwargs...) = isapprox(M, e, p; kwargs...)
function isapprox(M::DT, e::Identity, p; kwargs...) where {DT<:AbstractDecoratorManifold}
    isapprox(M.manifold, e, p; kwargs...)
    error("isapprox not implemented for manifold $(typeof(M)) and points $(typeof(e)) and $(typeof(p))")
end
function isapprox(M::Manifold, e::Identity, p; kwargs...) where {DT<:AbstractDecoratorManifold}
    error("isapprox not implemented for manifold $(typeof(M)) and points $(typeof(e)) and $(typeof(p))")
end
function isapprox(M::DT, e::E, ::E; kwargs...) where {E<:Identity, DT<:AbstractDecoratorManifold}
  return isapprox(M.manifold, e, e; kwargs...)
end
function isapprox(M::Manifold, e::E, ::E; kwargs...) where {E<:Identity}
    error("isapprox not implemented for manifold $(typeof(M)) and points $(typeof(e)) and $(typeof(e))")
end
function isapprox(G::GT, e::Identity{GT}, p; kwargs...) where {GT<:AbstractGroupManifold}
    return isapprox(G, identity(G, p), p; kwargs...)
end
function isapprox(
    ::GT,
    ::E,
    ::E;
    kwargs...,
) where {GT<:AbstractGroupManifold,E<:Identity{GT}}
    return true
end
function isapprox(G::GT, p, e::Identity{GT}; kwargs...) where {GT<:GroupManifold}
    return isapprox(G, e, p; kwargs...)
end
function isapprox(G::GT, e::Identity{GT}, p; kwargs...) where {GT<:GroupManifold}
    return isapprox(G, identity(G, p), p; kwargs...)
end
isapprox(::GT, ::E, ::E; kwargs...) where {GT<:GroupManifold,E<:Identity{GT}} = true

@doc raw"""
    compose(G::AbstractGroupManifold, p, q)

Compose elements $p,q ∈ \mathcal{G}$ using the group operation $p \circ q$.
"""
compose(M::DT, p, q) where {DT<:AbstractDecoratorManifold} = compose(M.manifold, p, q)
function compose(M::Manifold, p, q)
    return error("compose not implemented on $(typeof(M)) for elements $(typeof(p)) and $(typeof(q))")
end
function compose(G::AbstractGroupManifold, p, q)
    x = allocate_result(G, compose, p, q)
    return compose!(G, x, p, q)
end

compose!(M::DT, x, p, q) where {DT<:AbstractDecoratorManifold} = compose!(M.manifold, x, p, q)
function compose!(M::Manifold, x, p, q)
    return error("compose! not implemented on $(typeof(M)) for elements $(typeof(p)) and $(typeof(q))")
end

_action_order(p, q, conv::LeftAction) = (p, q)
_action_order(p, q, conv::RightAction) = (q, p)

@doc raw"""
    translate(G::AbstractGroupManifold, p, q)
    translate(G::AbstractGroupManifold, p, q, conv::ActionDirection=LeftAction()])

Translate group element $q$ by $p$ with the translation $τ_p$ with the specified
`conv`ention, either left ($L_p$) or right ($R_p$), defined as
```math
\begin{aligned}
L_p &: q ↦ p \circ q\\
R_p &: q ↦ q \circ p.
\end{aligned}
```
"""
translate(M::Manifold, p, q) = translate(M, p, q, LeftAction())
function translate(M::DT, p, q, conv::ActionDirection) where {DT<:AbstractDecoratorManifold}
    return translate(M.manifold, p, q, conv)
end
function translate(M::Manifold, p, q, conv::ActionDirection)
    return error("translate not implemented on $(typeof(M)) for elements $(typeof(p)) and $(typeof(q)) and direction $(typeof(conv))")
end
function translate(G::AbstractGroupManifold, p, q, conv::ActionDirection)
    return compose(G, _action_order(p, q, conv)...)
end

translate!(M::Manifold, x, p, q) = translate!(M, x, p, q, LeftAction())
function translate!(M::DT, x, p, q, conv::ActionDirection) where {DT<:AbstractDecoratorManifold}
    return translate!(M.manifold, x, p, q, conv)
end
function translate!(M::Manifold, x, p, q, conv::ActionDirection)
    return error("translate! not implemented on $(typeof(M)) for elements $(typeof(p)) and $(typeof(q)) and direction $(typeof(conv))")
end
function translate!(G::AbstractGroupManifold, x, p, q, conv::ActionDirection)
    return compose!(G, x, _action_order(p, q, conv)...)
end

@doc raw"""
    inverse_translate(G::AbstractGroupManifold, p, q)
    inverse_translate(G::AbstractGroupManifold, p, q, conv::ActionDirection=LeftAction())

Inverse translate group element $q$ by $p$ with the inverse translation $τ_p^{-1}$ with the
specified `conv`ention, either left ($L_p^{-1}$) or right ($R_p^{-1}$), defined as
```math
\begin{aligned}
L_p^{-1} &: q ↦ p^{-1} \circ q\\
R_p^{-1} &: q ↦ q \circ p^{-1}.
\end{aligned}
```
"""
inverse_translate(M::Manifold, p, q) = inverse_translate(M, p, q, LeftAction())
function inverse_translate(M::DT, p, q, conv::ActionDirection) where {DT<:AbstractDecoratorManifold}
    return inverse_translate(M.manifold, p, q, conv)
end
function inverse_translate(M::Manifold, p, q, conv::ActionDirection)
    return error("inverse_translate not implemented on $(typeof(M)) for elements $(typeof(p)) and $(typeof(q)) and direction $(typeof(conv))")
end
function inverse_translate(G::AbstractGroupManifold, p, q, conv::ActionDirection)
    return translate(G, inv(G, p), q, conv)
end

inverse_translate!(M::Manifold, x, p, q) = inverse_translate!(M, x, p, q, LeftAction())
function inverse_translate!(M::DT, x, p, q, conv::ActionDirection) where {DT<:AbstractDecoratorManifold}
    return inverse_translate!(M.manifold, x, p, q, conv)
end
function inverse_translate!(M::Manifold, x, p, q, conv::ActionDirection)
    return error("inverse_translate! not implemented on $(typeof(M)) for elements $(typeof(p)) and $(typeof(q)) and direction $(typeof(conv))")
end
function inverse_translate!(G::AbstractGroupManifold, x, p, q, conv::ActionDirection)
    return translate!(G, x, inv(G, p), q, conv)
end

@doc raw"""
    translate_diff(G::AbstractGroupManifold, p, q, X)
    translate_diff(G::AbstractGroupManifold, p, q, X, conv::ActionDirection=LeftAction())

For group elements $p, q ∈ \mathcal{G}$ and tangent vector $X ∈ T_q \mathcal{G}$, compute
the action of the differential of the translation $τ_p$ by $p$ on $X$, with the specified
left or right `conv`ention. The differential transports vectors:
```math
(\mathrm{d}τ_p)_q : T_q \mathcal{G} → T_{τ_p q} \mathcal{G}\\
```
"""
translate_diff(M::Manifold, p, q, X) = translate_diff(M, p, q, X, LeftAction())
function translate_diff(M::DT, p, q, X, conv::ActionDirection) where {DT<:AbstractDecoratorManifold}
    return translate_diff(M.manifold, p, q, X, conv)
end
function translate_diff(M::Manifold, p, q, X, conv::ActionDirection)
    return error("translate_diff not implemented on $(typeof(G)) for elements $(typeof(p)) and $(typeof(q)), vector $(typeof(X)), and direction $(typeof(conv))")
end
function translate_diff(G::AbstractGroupManifold, p, q, X, conv::ActionDirection)
    pq = translate(G, p, q, conv)
    Y = allocate_result(G, translate_diff, X, p, q)
    translate_diff!(G, Y, p, q, X, conv)
    return Y
end

function translate_diff!(M::Manifold, Y, p, q, X)
    return translate_diff!(M, Y, p, q, X, LeftAction())
end
function translate_diff!(M::DT, Y, p, q, X, conv::ActionDirection) where {DT<:AbstractDecoratorManifold}
    return translate_diff!(M.manifold, Y, p, q, X, conv)
end
function translate_diff!(M::Manifold, Y, p, q, X, conv::ActionDirection)
    return error("translate_diff! not implemented on $(typeof(M)) for elements $(typeof(Y)), $(typeof(p)) and $(typeof(q)), vector $(typeof(X)), and direction $(typeof(conv))")
end

@doc raw"""
    inverse_translate_diff(G::AbstractGroupManifold, p, q, X)
    inverse_translate_diff(G::AbstractGroupManifold, p, q, X, conv::ActionDirection=LeftAction())

For group elements $p, q ∈ \mathcal{G}$ and tangent vector $X ∈ T_q \mathcal{G}$, compute
the action on $X$ of the differential of the inverse translation $τ_p$ by $p$, with the
specified left or right `conv`ention. The differential transports vectors:
```math
(\mathrm{d}τ_p^{-1})_q : T_q \mathcal{G} → T_{τ_p^{-1} q} \mathcal{G}\\
```
"""
function inverse_translate_diff(M::Manifold, p, q, X)
    return inverse_translate_diff(M, p, q, X, LeftAction())
end
function inverse_translate_diff(M::DT, p, q, X, conv::ActionDirection) where {DT<:AbstractDecoratorManifold}
    return inverse_translate_diff(M.manifold, p, q, X, conv)
end
function inverse_translate_diff(M::Manifold, p, q, X, conv::ActionDirection)
    return error("inverse_translate_diff not implemented on $(typeof(M)) for elements $(typeof(p)) and $(typeof(q)), vector $(typeof(X)), and direction $(typeof(conv))")
end
function inverse_translate_diff(G::AbstractGroupManifold, p, q, X, conv::ActionDirection)
    return translate_diff(G, inv(G, p), q, X, conv)
end

function inverse_translate_diff!(M::Manifold, Y, p, q, X)
    return inverse_translate_diff!(M, Y, p, q, X, LeftAction())
end
function inverse_translate_diff!(M::DT, Y, p, q, X, conv::ActionDirection) where {DT<:AbstractDecoratorManifold}
    return inverse_translate_diff!(M.manifold, Y, p, q, X, conv)
end
function inverse_translate_diff!(
    M::Manifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection
)
    return error("inverse_translate_diff! not implemented on $(typeof(M)) for elements $(typeof(Y)), $(typeof(p)) and $(typeof(q)), vector $(typeof(X)), and direction $(typeof(conv))")
end
function inverse_translate_diff!(
    G::AbstractGroupManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return translate_diff!(G, Y, inv(G, p), q, X, conv)
end

@doc raw"""
    group_exp(G::AbstractGroupManifold, X)

Compute the group exponential of the Lie algebra element `X`.

Given an element $X ∈ 𝔤 = T_e \mathcal{G}$, where $e$ is the [`identity`](@ref) element of
the group $\mathcal{G}$, and $𝔤$ is its Lie algebra, the group exponential is the map

````math
\exp : 𝔤 → \mathcal{G},
````
such that for $t,s ∈ ℝ$, $γ(t) = \exp (t X)$ defines a one-parameter subgroup with the
following properties:

````math
\begin{aligned}
γ(t) &= γ(-t)^{-1}\\
γ(t + s) &= γ(t) \circ γ(s) = γ(s) \circ γ(t)\\
γ(0) &= e\\
\lim_{t → 0} \frac{d}{dt} γ(t) &= X.
\end{aligned}
````

!!! note
    In general, the group exponential map is distinct from the Riemannian exponential map
    [`exp`](@ref).

```
group_exp(G::AbstractGroupManifold{AdditionOperation}, X)
```

Compute $q = X$.

    group_exp(G::AbstractGroupManifold{MultiplicationOperation}, X)

For `Number` and `AbstractMatrix` types of `X`, compute the usual numeric/matrix
exponential,

````math
\exp X = \operatorname{Exp} X = \sum_{n=0}^∞ \frac{1}{n!} X^n.
````
"""
function group_exp(M::DT, X) where {DT<:AbstractDecoratorManifold}
    return group_exp(M.manifold, X)
end
function group_exp(M::Manifold, X)
    return error("group_exp not implemented on $(typeof(M)) for vector $(typeof(X)).")
end
function group_exp(G::AbstractGroupManifold, X)
    q = allocate_result(G, group_exp, X)
    return group_exp!(G, q, X)
end

function group_exp!(M::DT, q, X) where {DT<:AbstractDecoratorManifold}
    group_exp!(M.manifold, q, X)
end
function group_exp!(M::Manifold, q, X)
    return error("group_exp! not implemented on $(typeof(M)) for vector $(typeof(X)) and element $(typeof(q)).")
end

@doc raw"""
    group_log(G::AbstractGroupManifold, q)

Compute the group logarithm of the group element `q`.

Given an element $q ∈ \mathcal{G}$, compute the right inverse of the group exponential map
[`group_exp`](@ref), that is, the element $\log q = X ∈ 𝔤 = T_e \mathcal{G}$, such that
$q = \exp X$

!!! note
    In general, the group logarithm map is distinct from the Riemannian logarithm map
    [`log`](@ref).

```
group_log(G::AbstractGroupManifold{AdditionOperation}, q)
```

Compute $X = q$.

    group_log(G::AbstractGroupManifold{MultiplicationOperation}, q)

For `Number` and `AbstractMatrix` types of `q`, compute the usual numeric/matrix logarithm:

````math
\log q = \operatorname{Log} q = \sum_{n=1}^∞ \frac{(-1)^{n+1}}{n} (q - e)^n,
````

where $e$ here is the [`identity`](@ref) element, that is, $1$ for numeric $q$ or the
identity matrix $I_m$ for matrix $q ∈ ℝ^{m × m}$.
"""
function group_log(M::DT, q) where {DT<:AbstractDecoratorManifold}
    return group_log(M.manifold, q)
end
function group_log(M::Manifold, q)
    return error("group_log not implemented on $(typeof(M)) for element $(typeof(q)).")
end
function group_log(G::AbstractGroupManifold, q)
    X = allocate_result(G, group_log, q)
    return group_log!(G, X, q)
end

function group_log!(M::DT, X, q) where {DT<:AbstractDecoratorManifold}
    group_log!(M.manifold, X, q)
end
function group_log!(M::Manifold, X, q)
    return error("group_log! not implemented on $(typeof(M)) for element $(typeof(q)) and vector $(typeof(X)).")
end

############################
# Group-specific Retractions
############################

"""
    GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod

Retraction using the group exponential [`group_exp`](@ref) "translated" to any point on the
manifold.

For more details, see
[`retract`](@ref retract(::GroupManifold, p, X, ::GroupExponentialRetraction)).

# Constructor

    GroupExponentialRetraction(conv::ActionDirection = LeftAction())
"""
struct GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod end

function GroupExponentialRetraction(conv::ActionDirection = LeftAction())
    return GroupExponentialRetraction{typeof(conv)}()
end

"""
    GroupLogarithmicInverseRetraction{D<:ActionDirection} <: AbstractInverseRetractionMethod

Retraction using the group logarithm [`group_log`](@ref) "translated" to any point on the
manifold.

For more details, see
[`inverse_retract`](@ref inverse_retract(::GroupManifold, p, q ::GroupLogarithmicInverseRetraction)).

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

@doc raw"""
    retract(
        G::AbstractGroupManifold,
        p,
        X,
        method::GroupExponentialRetraction{<:ActionDirection},
    )

Compute the retraction using the group exponential [`group_exp`](@ref) "translated" to any
point on the manifold.
With a group translation ([`translate`](@ref)) $τ_p$ in a specified direction, the
retraction is

````math
\operatorname{retr}_p = τ_p \circ \exp \circ (\mathrm{d}τ_p^{-1})_p,
````

where $\exp$ is the group exponential ([`group_exp`](@ref)), and $(\mathrm{d}τ_p^{-1})_p$ is
the action of the differential of inverse translation $τ_p^{-1}$ evaluated at $p$ (see
[`inverse_translate_diff`](@ref)).
"""
retract(::GroupManifold, ::Any, ::Any, ::GroupExponentialRetraction)

function retract!(G::GroupManifold, q, p, X, method::GroupExponentialRetraction)
    return invoke(
        retract!,
        Tuple{Manifold,typeof(q),typeof(p),typeof(X),typeof(method)},
        G,
        q,
        p,
        X,
        method,
    )
end
function retract!(M::Manifold, q, p, X, method::GroupExponentialRetraction)
    conv = direction(method)
    Xₑ = inverse_translate_diff(M, p, p, X, conv)
    pinvq = group_exp(M, Xₑ)
    return translate!(M, q, p, pinvq, conv)
end

@doc raw"""
    inverse_retract(
        G::AbstractGroupManifold,
        p,
        X,
        method::GroupLogarithmicInverseRetraction{<:ActionDirection},
    )

Compute the inverse retraction using the group logarithm [`group_log`](@ref) "translated"
to any point on the manifold.
With a group translation ([`translate`](@ref)) $τ_p$ in a specified direction, the
retraction is

````math
\operatorname{retr}_p^{-1} = (\mathrm{d}τ_p)_e \circ \log \circ τ_p^{-1},
````

where $\log$ is the group logarithm ([`group_log`](@ref)), and $(\mathrm{d}τ_p)_e$ is the
action of the differential of translation $τ_p$ evaluated at the identity element $e$
(see [`translate_diff`](@ref)).
"""
inverse_retract(::GroupManifold, ::Any, ::Any, ::GroupLogarithmicInverseRetraction)

function inverse_retract!(
    G::GroupManifold,
    X,
    p,
    q,
    method::GroupLogarithmicInverseRetraction,
)
    return invoke(
        inverse_retract!,
        Tuple{Manifold,typeof(X),typeof(p),typeof(q),typeof(method)},
        G,
        X,
        p,
        q,
        method,
    )
end
function inverse_retract!(M::Manifold, X, p, q, method::GroupLogarithmicInverseRetraction)
    conv = direction(method)
    pinvq = inverse_translate(M, p, q, conv)
    Xₑ = group_log(M, pinvq)
    return translate_diff!(M, X, p, Identity(M), Xₑ, conv)
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
+(::Identity{G}, p) where {G<:AdditionGroup} = p
+(p, ::Identity{G}) where {G<:AdditionGroup} = p
+(e::E, ::E) where {G<:AdditionGroup,E<:Identity{G}} = e

-(e::Identity{G}) where {G<:AdditionGroup} = e
-(::Identity{G}, p) where {G<:AdditionGroup} = -p
-(p, ::Identity{G}) where {G<:AdditionGroup} = p
-(e::E, ::E) where {G<:AdditionGroup,E<:Identity{G}} = e

*(e::Identity{G}, p) where {G<:AdditionGroup} = e
*(p, e::Identity{G}) where {G<:AdditionGroup} = e
*(e::E, ::E) where {G<:AdditionGroup,E<:Identity{G}} = e

zero(e::Identity{G}) where {G<:AdditionGroup} = e

identity(::AdditionGroup, p) = zero(p)

identity!(::AdditionGroup, q, p) = fill!(q, 0)

inv(::AdditionGroup, p) = -p

inv!(::AdditionGroup, q, p) = copyto!(q, -p)

compose(::AdditionGroup, p, q) = p + q

function compose!(::GT, x, p, q) where {GT<:AdditionGroup}
    p isa Identity{GT} && return copyto!(x, q)
    q isa Identity{GT} && return copyto!(x, p)
    x .= p .+ q
    return x
end

translate_diff(::AdditionGroup, p, q, X, ::ActionDirection) = X

translate_diff!(::AdditionGroup, Y, p, q, X, ::ActionDirection) = copyto!(Y, X)

inverse_translate_diff(::AdditionGroup, p, q, X, ::ActionDirection) = X

function inverse_translate_diff!(::AdditionGroup, Y, p, q, X, ::ActionDirection)
    return copyto!(Y, X)
end

group_exp(::AdditionGroup, X) = X

group_exp!(::AdditionGroup, q, X) = copyto!(q, X)

group_log(::AdditionGroup, q) = q

group_log!(::AdditionGroup, X, q) = copyto!(X, q)

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
*(::Identity{G}, p) where {G<:MultiplicationGroup} = p
*(p, ::Identity{G}) where {G<:MultiplicationGroup} = p
*(e::E, ::E) where {G<:MultiplicationGroup,E<:Identity{G}} = e
*(::Identity{<:MultiplicationGroup}, e::Identity{<:AdditionGroup}) = e

/(p, ::Identity{G}) where {G<:MultiplicationGroup} = p
/(::Identity{G}, p) where {G<:MultiplicationGroup} = inv(p)
/(e::E, ::E) where {G<:MultiplicationGroup,E<:Identity{G}} = e

\(p, ::Identity{G}) where {G<:MultiplicationGroup} = inv(p)
\(::Identity{G}, p) where {G<:MultiplicationGroup} = p
\(e::E, ::E) where {G<:MultiplicationGroup,E<:Identity{G}} = e

inv(e::Identity{G}) where {G<:MultiplicationGroup} = e

one(e::Identity{G}) where {G<:MultiplicationGroup} = e

transpose(e::Identity{G}) where {G<:MultiplicationGroup} = e

LinearAlgebra.det(::Identity{<:MultiplicationGroup}) = 1

LinearAlgebra.mul!(q, e::Identity{G}, p) where {G<:MultiplicationGroup} = copyto!(q, p)
LinearAlgebra.mul!(q, p, e::Identity{G}) where {G<:MultiplicationGroup} = copyto!(q, p)
function LinearAlgebra.mul!(q, e::E, ::E) where {G<:MultiplicationGroup,E<:Identity{G}}
    return identity!(e.group, q, e)
end

identity(::MultiplicationGroup, p) = one(p)

function identity!(G::GT, q, p) where {GT<:MultiplicationGroup}
    isa(p, Identity{GT}) || return copyto!(q, one(p))
    error("identity! not implemented on $(typeof(G)) for points $(typeof(q)) and $(typeof(p))")
end
identity!(::MultiplicationGroup, q::AbstractMatrix, p) = copyto!(q, I)

inv(::MultiplicationGroup, p) = inv(p)

inv!(G::MultiplicationGroup, q, p) = copyto!(q, inv(G, p))

compose(::MultiplicationGroup, p, q) = p * q

# TODO: x might alias with p or q, we might be able to optimize it if it doesn't.
compose!(::MultiplicationGroup, x, p, q) = copyto!(x, p * q)

inverse_translate(::MultiplicationGroup, p, q, ::LeftAction) = p \ q
inverse_translate(::MultiplicationGroup, p, q, ::RightAction) = q / p

function inverse_translate!(G::MultiplicationGroup, x, p, q, conv::ActionDirection)
    return copyto!(x, inverse_translate(G, p, q, conv))
end

function group_exp!(G::MultiplicationGroup, q, X)
    X isa Union{Number,AbstractMatrix} && return copyto!(q, exp(X))
    return error("group_exp! not implemented on $(typeof(G)) for vector $(typeof(X)) and element $(typeof(q)).")
end
