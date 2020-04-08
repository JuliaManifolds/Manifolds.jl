@doc raw"""
    AbstractGroupOperation

Abstract type for smooth binary operations $∘$ on elements of a Lie group $\mathcal{G}$:
```math
∘ : \mathcal{G} × \mathcal{G} → \mathcal{G}
```
An operation can be either defined for a specific [`AbstractGroupManifold`](@ref) over
number system `𝔽` or in general, by defining for an operation `Op` the following methods:

    identity!(::AbstractGroupManifold{𝔽,Op}, q, q)
    identity(::AbstractGroupManifold{𝔽,Op}, p)
    inv!(::AbstractGroupManifold{𝔽,Op}, q, p)
    inv(::AbstractGroupManifold{𝔽,Op}, p)
    compose(::AbstractGroupManifold{𝔽,Op}, p, q)
    compose!(::AbstractGroupManifold{𝔽,Op}, x, p, q)

Note that a manifold is connected with an operation by wrapping it with a decorator,
[`AbstractGroupManifold`](@ref). In typical cases the concrete wrapper
[`GroupManifold`](@ref) can be used.
"""
abstract type AbstractGroupOperation end

@doc raw"""
    AbstractGroupManifold{𝔽,O<:AbstractGroupOperation} <: AbstractDecoratorManifold{𝔽}

Abstract type for a Lie group, a group that is also a smooth manifold with an
[`AbstractGroupOperation`](@ref), a smooth binary operation. `AbstractGroupManifold`s must
implement at least [`inv`](@ref), [`identity`](@ref), [`compose`](@ref), and
[`translate_diff`](@ref).
"""
abstract type AbstractGroupManifold{𝔽,O<:AbstractGroupOperation} <:
              AbstractDecoratorManifold{𝔽} end

"""
    GroupManifold{𝔽,M<:Manifold{𝔽},O<:AbstractGroupOperation} <: AbstractGroupManifold{𝔽,O}

Decorator for a smooth manifold that equips the manifold with a group operation, thus making
it a Lie group. See [`AbstractGroupManifold`](@ref) for more details.

Group manifolds by default forward metric-related operations to the wrapped manifold.

# Constructor

    GroupManifold(manifold, op)
"""
struct GroupManifold{𝔽,M<:Manifold{𝔽},O<:AbstractGroupOperation} <:
       AbstractGroupManifold{𝔽,O}
    manifold::M
    op::O
end

show(io::IO, G::GroupManifold) = print(io, "GroupManifold($(G.manifold), $(G.op))")

const GROUP_MANIFOLD_BASIS_DISAMBIGUATION =
    [AbstractDecoratorManifold, ValidationManifold, VectorBundle]

"""
    base_group(M::Manifold) -> AbstractGroupManifold

Un-decorate `M` until an `AbstractGroupManifold` is encountered.
Return an error if the [`base_manifold`](@ref) is reached without encountering a group.
"""
base_group(M::AbstractDecoratorManifold) = base_group(decorated_manifold(M))
function base_group(M::Manifold)
    error("base_group: no base group found.")
end
base_group(G::AbstractGroupManifold) = G

decorator_group_dispatch(M::Manifold) = Val(false)
function decorator_group_dispatch(M::AbstractDecoratorManifold)
    return decorator_group_dispatch(decorated_manifold(M))
end
decorator_group_dispatch(M::AbstractGroupManifold) = Val(true)

function is_group_decorator(M::Manifold)
    return _extract_val(decorator_group_dispatch(M))
end

default_decorator_dispatch(::AbstractGroupManifold) = Val(false)

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
    Identity(G::AbstractGroupManifold, p)

The group identity element $e ∈ \mathcal{G}$ represented by point `p`.
"""
struct Identity{G<:AbstractGroupManifold,PT}
    group::G
    p::PT
end

Identity(M::AbstractDecoratorManifold, p) = Identity(decorated_manifold(M), p)
function Identity(M::Manifold, p)
    error("Identity not implemented for manifold $(M) and point $(p).")
end

function ==(e1::Identity, e2::Identity)
    return e1.p == e2.p && e1.group == e2.group
end

make_identity(M::Manifold, p) = Identity(M, identity(M, p))

show(io::IO, e::Identity) = print(io, "Identity($(e.group), $(e.p))")

# To ensure allocate_result_type works
number_eltype(e::Identity) = Bool

copyto!(e::TE, ::TE) where {TE<:Identity} = e
copyto!(p, ::TE) where {TE<:Identity} = copyto!(p, e.p)
copyto!(p::AbstractArray, e::TE) where {TE<:Identity} = copyto!(p, e.p)

isapprox(p, e::Identity; kwargs...) = isapprox(e::Identity, p; kwargs...)
isapprox(e::Identity, p; kwargs...) = isapprox(e.group, e, p; kwargs...)
isapprox(e::E, ::E; kwargs...) where {E<:Identity} = true

function allocate_result(
    M::Manifold,
    f::typeof(get_coordinates),
    e::Identity,
    X,
    B::AbstractBasis,
)
    T = allocate_result_type(M, f, (e.p, X))
    return allocate(e.p, T, Size(number_of_coordinates(M, B)))
end

function decorator_transparent_dispatch(
    ::typeof(get_vector),
    ::AbstractGroupManifold,
    args...,
)
    return Val(:parent)
end
function allocate_result(M::Manifold, f::typeof(get_vector), e::Identity, Xⁱ)
    is_group_decorator(M) && return allocate_result(base_group(M), f, e, Xⁱ)
    error("allocate_result not implemented for manifold $(M), function $(f), point $(e), and vector $(Xⁱ).")
end
function allocate_result(M::AbstractGroupManifold, f::typeof(get_vector), e::Identity, Xⁱ)
    error("allocate_result not implemented for group manifold $(M), function $(f), $(e), and vector $(Xⁱ).")
end
function allocate_result(
    G::GT,
    ::typeof(get_vector),
    ::Identity{GT},
    Xⁱ,
) where {GT<:AbstractGroupManifold}
    B = VectorBundleFibers(TangentSpace, G)
    return allocate(Xⁱ, Size(representation_size(B)))
end

function decorator_transparent_dispatch(
    ::typeof(get_coordinates),
    ::AbstractGroupManifold,
    args...,
)
    return Val(:parent)
end
function allocate_result(
    M::AbstractDecoratorManifold,
    f::typeof(get_coordinates),
    e::Identity,
    X,
)
    is_group_decorator(M) && return allocate_result(base_group(M), f, e, X)
    error("allocate_result not implemented for manifold $(M), function $(f), point $(e), and vector $(X).")
end
function allocate_result(
    M::AbstractGroupManifold,
    f::typeof(get_coordinates),
    e::Identity,
    X,
)
    error("allocate_result not implemented for group manifold $(M), function $(f), $(e), and vector $(X).")
end
function allocate_result(
    G::GT,
    ::typeof(get_coordinates),
    ::Identity{GT},
    X,
) where {GT<:AbstractGroupManifold}
    return allocate(X, Size(manifold_dimension(G)))
end

function get_vector(M::AbstractGroupManifold, e::Identity, X, B::VeeOrthogonalBasis)
    M != e.group && error("On $(M) the identity $(e) does not match to perform get_vector.")
    return get_vector(decorated_manifold(M), e.p, X, B)
end
function get_vector(M::Manifold, e::Identity, X, B::VeeOrthogonalBasis)
    M != e.group.manifold &&
    error("On $(M) the identity $(e) does not match to perform get_vector.")
    return get_vector(M, e.p, X, B)
end
for MT in GROUP_MANIFOLD_BASIS_DISAMBIGUATION
    eval(
        quote
            @invoke_maker 1 Manifold get_vector(
                M::$MT,
                e::Identity,
                X,
                B::VeeOrthogonalBasis,
            )
        end,
    )
end
function get_vector!(M::AbstractGroupManifold, Y, e::Identity, X, B::VeeOrthogonalBasis)
    M != e.group && error("On $(M) the identity $(e) does not match to perform get_vector!")
    return get_vector!(decorated_manifold(M), Y, e.p, X, B)
end
function get_vector!(M::Manifold, Y, e::Identity, X, B::VeeOrthogonalBasis)
    M != e.group.manifold &&
    error("On $(M) the identity $(e) does not match to perform get_vector!")
    return get_vector!(M, Y, e.p, X, B)
end
for MT in GROUP_MANIFOLD_BASIS_DISAMBIGUATION
    eval(quote
        @invoke_maker 1 Manifold get_vector!(
            M::$MT,
            Y,
            e::Identity,
            X,
            B::VeeOrthogonalBasis,
        )
    end)
end

function get_coordinates(M::AbstractGroupManifold, e::Identity, X, B::VeeOrthogonalBasis)
    M != e.group &&
    error("On $(M) the identity $(e) does not match to perform get_coordinates")
    return get_coordinates(decorated_manifold(M), e.p, X, B)
end
function get_coordinates(M::Manifold, e::Identity, X, B::VeeOrthogonalBasis)
    M != e.group.manifold &&
    error("On $(M) the identity $(e) does not match to perform get_coordinates")
    return get_coordinates(M, e.p, X, B)
end
for MT in GROUP_MANIFOLD_BASIS_DISAMBIGUATION
    eval(quote
        @invoke_maker 1 Manifold get_coordinates(
            M::$MT,
            e::Identity,
            X,
            B::VeeOrthogonalBasis,
        )
    end)
end

function get_coordinates!(
    M::AbstractGroupManifold,
    Y,
    e::Identity,
    X,
    B::VeeOrthogonalBasis,
)
    M != e.group &&
    error("On $(M) the identity $(e) does not match to perform get_coordinates!")
    return get_coordinates!(decorated_manifold(M), Y, e.p, X, B)
end
function get_coordinates!(M::Manifold, Y, e::Identity, X, B::VeeOrthogonalBasis)
    M != e.group.manifold &&
    error("On $(M) the identity $(e) does not match to perform get_coordinates!")
    return get_coordinates!(M, Y, e.p, X, B)
end
for MT in GROUP_MANIFOLD_BASIS_DISAMBIGUATION
    eval(quote
        @invoke_maker 1 Manifold get_coordinates!(
            M::$MT,
            Y,
            e::Identity,
            X,
            B::VeeOrthogonalBasis,
        )
    end)
end

import ManifoldsBase.check_manifold_point__transparent
@decorator_transparent_fallback :transparent function check_manifold_point(
    G::AbstractGroupManifold,
    e::Identity;
    kwargs...,
)
    return DomainError(e, "The identity element $(e) does not belong to $(G).")
end
@decorator_transparent_fallback :transparent function check_manifold_point(
    G::GT,
    e::Identity{GT};
    kwargs...,
) where {GT<:AbstractGroupManifold}
    return nothing
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
inv(::AbstractGroupManifold, ::Any...)
@decorator_transparent_function :intransparent function inv(G::AbstractGroupManifold, p)
    q = allocate_result(G, inv, p)
    return inv!(G, q, p)
end

@decorator_transparent_function :intransparent function inv!(G::AbstractGroupManifold, q, p)
    inv!(G.manifold, q, p)
end

@doc raw"""
    identity(G::AbstractGroupManifold, p)

Identity element $e ∈ \mathcal{G}$, such that for any element $p ∈ \mathcal{G}$,
$p \circ e = e \circ p = p$.
The returned element is of a similar type to `p`.
"""
identity(::AbstractGroupManifold, ::Any)
@decorator_transparent_signature identity(G::AbstractDecoratorManifold, p)
function identity(G::AbstractGroupManifold, p)
    y = allocate_result(G, identity, p)
    return identity!(G, y, p)
end

@decorator_transparent_signature identity!(G::AbstractDecoratorManifold, q, p)
function decorator_transparent_dispatch(::typeof(identity!), ::AbstractGroupManifold, q, p)
    return Val(:intransparent)
end

function isapprox(G::GT, e::Identity{GT}, p; kwargs...) where {GT<:AbstractGroupManifold}
    return isapprox(G, e.p, p; kwargs...)
end
function isapprox(G::GT, p, e::Identity{GT}; kwargs...) where {GT<:AbstractGroupManifold}
    return isapprox(G, e, p; kwargs...)
end
isapprox(::GT, ::E, ::E; kwargs...) where {GT<:AbstractGroupManifold,E<:Identity{GT}} = true

function decorator_transparent_dispatch(
    ::typeof(isapprox),
    M::AbstractDecoratorManifold,
    e::E,
    p,
) where {E<:Identity{<:AbstractGroupManifold}}
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(isapprox),
    M::AbstractDecoratorManifold,
    p,
    e::E,
) where {E<:Identity{<:AbstractGroupManifold}}
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(isapprox),
    M::AbstractDecoratorManifold,
    e1::E1,
    e2::E2,
) where {E1<:Identity{<:AbstractGroupManifold},E2<:Identity{<:AbstractGroupManifold}}
    return Val(:transparent)
end

@doc raw"""
    compose(G::AbstractGroupManifold, p, q)

Compose elements $p,q ∈ \mathcal{G}$ using the group operation $p \circ q$.
"""
compose(::AbstractGroupManifold, ::Any...)
@decorator_transparent_signature compose(M::AbstractDecoratorManifold, p, q)
function compose(G::AbstractGroupManifold, p, q)
    x = allocate_result(G, compose, p, q)
    return compose!(G, x, p, q)
end
function decorator_transparent_dispatch(
    ::typeof(compose),
    M::AbstractGroupManifold,
    x,
    p,
    q,
)
    return Val(:intransparent)
end

@decorator_transparent_signature compose!(M::AbstractDecoratorManifold, x, p, q)
function decorator_transparent_dispatch(
    ::typeof(compose!),
    M::AbstractGroupManifold,
    x,
    p,
    q,
)
    return Val(:intransparent)
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
translate(::AbstractGroupManifold, ::Any...)
@decorator_transparent_signature translate(M::AbstractDecoratorManifold, p, q)
function translate(G::AbstractGroupManifold, p, q)
    return translate(G, p, q, LeftAction())
end
@decorator_transparent_signature translate(
    M::AbstractDecoratorManifold,
    p,
    q,
    conv::ActionDirection,
)
function translate(G::AbstractGroupManifold, p, q, conv::ActionDirection)
    return compose(G, _action_order(p, q, conv)...)
end

function decorator_transparent_dispatch(
    ::typeof(translate),
    ::AbstractGroupManifold,
    args...,
)
    return Val(:intransparent)
end

@decorator_transparent_signature translate!(M::AbstractDecoratorManifold, X, p, q)
function translate!(G::AbstractGroupManifold, X, p, q)
    return translate!(G, X, p, q, LeftAction())
end
@decorator_transparent_signature translate!(
    M::AbstractDecoratorManifold,
    X,
    p,
    q,
    conv::ActionDirection,
)
function translate!(G::AbstractGroupManifold, X, p, q, conv::ActionDirection)
    return compose!(G, X, _action_order(p, q, conv)...)
end

function decorator_transparent_dispatch(
    ::typeof(translate!),
    ::AbstractGroupManifold,
    args...,
)
    return Val(:intransparent)
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
inverse_translate(::AbstractGroupManifold, ::Any...)
@decorator_transparent_function :intransparent function inverse_translate(
    G::AbstractGroupManifold,
    p,
    q,
)
    return inverse_translate(G, p, q, LeftAction())
end
@decorator_transparent_function :intransparent function inverse_translate(
    G::AbstractGroupManifold,
    p,
    q,
    conv::ActionDirection,
)
    return translate(G, inv(G, p), q, conv)
end

@decorator_transparent_function :intransparent function inverse_translate!(
    G::AbstractGroupManifold,
    X,
    p,
    q,
)
    return inverse_translate!(G, X, p, q, LeftAction())
end
@decorator_transparent_function :intransparent function inverse_translate!(
    G::AbstractGroupManifold,
    X,
    p,
    q,
    conv::ActionDirection,
)
    return translate!(G, X, inv(G, p), q, conv)
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
translate_diff(::AbstractGroupManifold, ::Any...)
@decorator_transparent_signature translate_diff(M::AbstractDecoratorManifold, p, q, X)
function translate_diff(G::AbstractGroupManifold, p, q, X)
    return translate_diff(G, p, q, X, LeftAction())
end
@decorator_transparent_signature translate_diff(
    M::AbstractDecoratorManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
function translate_diff(G::AbstractGroupManifold, p, q, X, conv::ActionDirection)
    Y = allocate_result(G, translate_diff, X, p, q)
    translate_diff!(G, Y, p, q, X, conv)
    return Y
end

@decorator_transparent_signature translate_diff!(M::AbstractDecoratorManifold, Y, p, q, X)
function translate_diff!(G::AbstractGroupManifold, Y, p, q, X)
    return translate_diff!(G, Y, p, q, X, LeftAction())
end
@decorator_transparent_signature translate_diff!(
    M::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection,
)
function decorator_transparent_dispatch(
    ::typeof(translate_diff!),
    G::AbstractGroupManifold,
    Y,
    p,
    q,
    X,
    conv,
)
    return Val(:intransparent)
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
inverse_translate_diff(::AbstractGroupManifold, ::Any...)
@decorator_transparent_function :intransparent function inverse_translate_diff(
    G::AbstractGroupManifold,
    p,
    q,
    X,
)
    return inverse_translate_diff(G, p, q, X, LeftAction())
end
@decorator_transparent_function :intransparent function inverse_translate_diff(
    G::AbstractGroupManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return translate_diff(G, inv(G, p), q, X, conv)
end

@decorator_transparent_function :intransparent function inverse_translate_diff!(
    G::AbstractGroupManifold,
    Y,
    p,
    q,
    X,
)
    return inverse_translate_diff!(G, Y, p, q, X, LeftAction())
end
@decorator_transparent_function :intransparent function inverse_translate_diff!(
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
group_exp(G::AbstractGroupManifold{𝔽,AdditionOperation}, X) where {𝔽}
```

Compute $q = X$.

    group_exp(G::AbstractGroupManifold{𝔽,MultiplicationOperation}, X) where {𝔽}

For `Number` and `AbstractMatrix` types of `X`, compute the usual numeric/matrix
exponential,

````math
\exp X = \operatorname{Exp} X = \sum_{n=0}^∞ \frac{1}{n!} X^n.
````
"""
group_exp(::AbstractGroupManifold, ::Any...)
@decorator_transparent_signature group_exp(G::AbstractDecoratorManifold, X)
function group_exp(G::AbstractGroupManifold, X)
    q = allocate_result(G, group_exp, X)
    return group_exp!(G, q, X)
end

@decorator_transparent_signature group_exp!(G::AbstractDecoratorManifold, q, X)

function decorator_transparent_dispatch(::typeof(group_exp), ::AbstractGroupManifold, q, X)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(::typeof(group_exp!), ::AbstractGroupManifold, q, X)
    return Val(:intransparent)
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
group_log(G::AbstractGroupManifold{𝔽,AdditionOperation}, q) where {𝔽}
```

Compute $X = q$.

    group_log(G::AbstractGroupManifold{𝔽,MultiplicationOperation}, q) where {𝔽}

For `Number` and `AbstractMatrix` types of `q`, compute the usual numeric/matrix logarithm:

````math
\log q = \operatorname{Log} q = \sum_{n=1}^∞ \frac{(-1)^{n+1}}{n} (q - e)^n,
````

where $e$ here is the [`identity`](@ref) element, that is, $1$ for numeric $q$ or the
identity matrix $I_m$ for matrix $q ∈ ℝ^{m × m}$.
"""
group_log(::AbstractGroupManifold, ::Any...)
@decorator_transparent_signature group_log(M::AbstractDecoratorManifold, q)
function group_log(G::AbstractGroupManifold, q)
    X = allocate_result(G, group_log, q)
    return group_log!(G, X, q)
end
function decorator_transparent_dispatch(::typeof(group_log), ::AbstractGroupManifold, X, q)
    return Val(:intransparent)
end

@decorator_transparent_signature group_log!(M::AbstractDecoratorManifold, X, q)
function decorator_transparent_dispatch(::typeof(group_log!), ::AbstractGroupManifold, X, q)
    return Val(:intransparent)
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
function retract(G::AbstractGroupManifold, p, X, method::GroupExponentialRetraction)
    conv = direction(method)
    Xₑ = inverse_translate_diff(G, p, p, X, conv)
    pinvq = group_exp(G, Xₑ)
    q = translate(G, p, pinvq, conv)
    return q
end

function retract!(G::AbstractGroupManifold, q, p, X, method::GroupExponentialRetraction)
    conv = direction(method)
    Xₑ = inverse_translate_diff(G, p, p, X, conv)
    pinvq = group_exp(G, Xₑ)
    return translate!(G, q, p, pinvq, conv)
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
function inverse_retract(G::GroupManifold, p, q, method::GroupLogarithmicInverseRetraction)
    conv = direction(method)
    pinvq = inverse_translate(G, p, q, conv)
    Xₑ = group_log(G, pinvq)
    return translate_diff(G, p, Identity(G, p), Xₑ, conv)
end

function inverse_retract!(
    G::AbstractGroupManifold,
    X,
    p,
    q,
    method::GroupLogarithmicInverseRetraction,
)
    conv = direction(method)
    pinvq = inverse_translate(G, p, q, conv)
    Xₑ = group_log(G, pinvq)
    return translate_diff!(G, X, p, Identity(G, p), Xₑ, conv)
end

#################################
# Overloads for AdditionOperation
#################################

"""
    AdditionOperation <: AbstractGroupOperation

Group operation that consists of simple addition.
"""
struct AdditionOperation <: AbstractGroupOperation end

const AdditionGroup = AbstractGroupManifold{𝔽,AdditionOperation} where {𝔽}

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

const MultiplicationGroup = AbstractGroupManifold{𝔽,MultiplicationOperation} where {𝔽}

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
