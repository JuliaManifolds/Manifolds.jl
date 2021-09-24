@doc raw"""
    AbstractGroupOperation

Abstract type for smooth binary operations $∘$ on elements of a Lie group $\mathcal{G}$:
```math
∘ : \mathcal{G} × \mathcal{G} → \mathcal{G}
```
An operation can be either defined for a specific [`AbstractGroupManifold`](@ref) over
number system `𝔽` or in general, by defining for an operation `Op` the following methods:

    identity_element!(::AbstractGroupManifold{𝔽,Op}, q, q)
    inv!(::AbstractGroupManifold{𝔽,Op}, q, p)
    _compose!(::AbstractGroupManifold{𝔽,Op}, x, p, q)

Note that a manifold is connected with an operation by wrapping it with a decorator,
[`AbstractGroupManifold`](@ref). In typical cases the concrete wrapper
[`GroupManifold`](@ref) can be used.
"""
abstract type AbstractGroupOperation end

"""
    abstract type AbstractGroupDecroatorType <: AbstractDecoratorType

A common decorator type for all group decorators.
It is similar to [`DefaultEmbeddingType`](@ref) but for groups.
"""
abstract type AbstractGroupDecoratorType <: AbstractDecoratorType end

"""
    struct DefaultGroupDecoratorType <: AbstractDecoratorType

The default group decorator type with no special properties.
"""
struct DefaultGroupDecoratorType <: AbstractGroupDecoratorType end
"""
    struct TransparentGroupDecoratorType <: AbstractDecoratorType

A transparent group decorator type that acts transparently, similar to
the [`TransparentIsometricEmbedding`](@ref), i.e. it passes through all metric-related functions such as
logarithmic and exponential map as well as retraction and inverse retractions
to the manifold it decorates.
"""
struct TransparentGroupDecoratorType <: AbstractGroupDecoratorType end

@doc raw"""
    AbstractGroupManifold{𝔽,O<:AbstractGroupOperation} <: AbstractDecoratorManifold{𝔽}

Abstract type for a Lie group, a group that is also a smooth manifold with an
[`AbstractGroupOperation`](@ref), a smooth binary operation. `AbstractGroupManifold`s must
implement at least [`inv`](@ref), [`compose`](@ref), and
[`translate_diff`](@ref).
"""
abstract type AbstractGroupManifold{𝔽,O<:AbstractGroupOperation,T<:AbstractDecoratorType} <:
              AbstractDecoratorManifold{𝔽,T} end

"""
    GroupManifold{𝔽,M<:AbstractManifold{𝔽},O<:AbstractGroupOperation} <: AbstractGroupManifold{𝔽,O}

Decorator for a smooth manifold that equips the manifold with a group operation, thus making
it a Lie group. See [`AbstractGroupManifold`](@ref) for more details.

Group manifolds by default forward metric-related operations to the wrapped manifold.

# Constructor

    GroupManifold(manifold, op)
"""
struct GroupManifold{𝔽,M<:AbstractManifold{𝔽},O<:AbstractGroupOperation} <:
       AbstractGroupManifold{𝔽,O,TransparentGroupDecoratorType}
    manifold::M
    op::O
end

Base.show(io::IO, G::GroupManifold) = print(io, "GroupManifold($(G.manifold), $(G.op))")

const GROUP_MANIFOLD_BASIS_DISAMBIGUATION =
    [AbstractDecoratorManifold, ValidationManifold, VectorBundle]

"""
    base_group(M::AbstractManifold) -> AbstractGroupManifold

Un-decorate `M` until an `AbstractGroupManifold` is encountered.
Return an error if the [`base_manifold`](@ref) is reached without encountering a group.
"""
base_group(M::AbstractDecoratorManifold) = base_group(decorated_manifold(M))
function base_group(::AbstractManifold)
    return error("base_group: no base group found.")
end
base_group(G::AbstractGroupManifold) = G

"""
    base_manifold(M::AbstractGroupManifold, d::Val{N} = Val(-1))

Return the base manifold of `M` that is enhanced with its group.
While functions like `inner` might be overwritten to use the (decorated) manifold
representing the group, the `base_manifold` is the manifold itself.
Hence for this abstract case, just `M` is returned.
"""
base_manifold(M::AbstractGroupManifold, ::Val{N}=Val(-1)) where {N} = M

"""
    base_manifold(M::GroupManifold, d::Val{N} = Val(-1))

Return the base manifold of `M` that is enhanced with its group.
Here, the internally stored enhanced manifold `M.manifold` is returned.
"""
base_manifold(G::GroupManifold, ::Val{N}=Val(-1)) where {N} = G.manifold

decorator_group_dispatch(::AbstractManifold) = Val(false)
function decorator_group_dispatch(M::AbstractDecoratorManifold)
    return decorator_group_dispatch(decorated_manifold(M))
end
decorator_group_dispatch(::AbstractGroupManifold) = Val(true)

function is_group_decorator(M::AbstractManifold)
    return _extract_val(decorator_group_dispatch(M))
end

default_decorator_dispatch(::AbstractGroupManifold) = Val(false)

(op::AbstractGroupOperation)(M::AbstractManifold) = GroupManifold(M, op)
function (::Type{T})(M::AbstractManifold) where {T<:AbstractGroupOperation}
    return GroupManifold(M, T())
end

manifold_dimension(G::GroupManifold) = manifold_dimension(G.manifold)

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
    Identity{O<:AbstractGroupOperation}

Represent the group identity element ``e ∈ \mathcal{G}`` on an [`AbstractGroupManifold`](@ref) `G`
with [`AbstractGroupOperation`](@ref) of type `O`.

Similar to the philosophy that points are agnostic of their group at hand, the identity
does not store the group `g` it belongs to. However it depends on the type of the [`AbstractGroupOperation`](@ref) used.

See also [`identity_element`](@ref) on how to obtain the corresponding [`AbstractManifoldPoint`](@ref) or array representation.

# Constructors

    Identity(G::AbstractGroupManifold{𝔽,O})
    Identity(o::O)
    Identity(::Type{O})

create the identity of the corresponding subtype `O<:`[`AbstractGroupOperation`](@ref)
"""
struct Identity{O<:AbstractGroupOperation} end

function Identity(::AbstractGroupManifold{𝔽,O}) where {𝔽,O<:AbstractGroupOperation}
    return Identity{O}()
end
Identity(M::AbstractDecoratorManifold) = Identity(base_group(M))
Identity(::O) where {O<:AbstractGroupOperation} = Identity(O)
Identity(::Type{O}) where {O<:AbstractGroupOperation} = Identity{O}()

# To ensure allocate_result_type works in general if idenitty apears in the tuple
number_eltype(::Identity) = Bool

@doc raw"""
    identity_element(G::AbstractGroupManifold)

Return a point representation of the [`Identity`](@ref) on the [`AbstractGroupManifold`](@ref) `G`.
By default this representation is the default array or number representation.
It should return the corresponding [`AbstractManifoldPoint`](@ref) of points on `G` if
points are not represented by arrays.
"""
identity_element(G::AbstractGroupManifold)
@decorator_transparent_function function identity_element(G::AbstractGroupManifold)
    q = allocate_result(G, identity_element)
    return identity_element!(G, q)
end

@decorator_transparent_signature identity_element!(G::AbstractGroupManifold, p)

function allocate_result(G::AbstractGroupManifold, ::typeof(identity_element))
    return zeros(representation_size(G)...)
end

@doc raw"""
    identity_element(G::AbstractGroupManifold, p)

Return a point representation of the [`Identity`](@ref) on the [`AbstractGroupManifold`](@ref) `G`,
where `p` indicates the type to represent the identity.
"""
identity_element(G::AbstractGroupManifold, p)
@decorator_transparent_function function identity_element(G::AbstractGroupManifold, p)
    q = allocate_result(G, identity_element, p)
    return identity_element!(G, q)
end

@doc raw"""
    identity_element!(G::AbstractGroupManifold, p)

Return a point representation of the [`Identity`](@ref) on the [`AbstractGroupManifold`](@ref) `G`
in place of `p`.
"""
identity_element!(G::AbstractGroupManifold, p)

@doc raw"""
    is_identity(G, q; kwargs)

Check whether `q` is the identity on the [`AbstractGroupManifold`](@ref) `G`, i.e. it is either
the [`Identity`](@ref)`{O}` with the corresponding [`AbstractGroupOperation`](@ref) `O`, or
(approximately) the correct point representation.
"""
is_identity(G::AbstractGroupManifold, q)

@decorator_transparent_function function is_identity(G::AbstractGroupManifold, q; kwargs...)
    return isapprox(G, identity_element(G), q; kwargs...)
end
function is_identity(
    ::AbstractGroupManifold{𝔽,O},
    ::Identity{O};
    kwargs...,
) where {𝔽,O<:AbstractGroupOperation}
    return true
end
is_identity(::AbstractGroupManifold, ::Identity; kwargs...) = false

function isapprox(
    G::AbstractGroupManifold{𝔽,O},
    p::Identity{O},
    q;
    kwargs...,
) where {𝔽,O<:AbstractGroupOperation}
    return is_identity(G, q; kwargs...)
end
function isapprox(
    G::AbstractGroupManifold{𝔽,O},
    p,
    q::Identity{O};
    kwargs...,
) where {𝔽,O<:AbstractGroupOperation}
    return is_identity(G, p; kwargs...)
end
function isapprox(
    G::AbstractGroupManifold{𝔽,O},
    p::Identity{O},
    q::Identity{O};
    kwargs...,
) where {𝔽,O<:AbstractGroupOperation}
    return true
end
function isapprox(
    G::AbstractGroupManifold{𝔽,O},
    p::Identity{O},
    X,
    Y;
    kwargs...,
) where {𝔽,O<:AbstractGroupOperation}
    return isapprox(G, identity_element(G), X, Y; kwargs...)
end
Base.isapprox(::AbstractGroupManifold, ::Identity, ::Identity; kwargs...) = false

function Base.show(io::IO, ::Identity{O}) where {O<:AbstractGroupOperation}
    return print(io, "Identity($O)")
end

function check_point(
    G::AbstractGroupManifold{𝔽,O},
    e::Identity{O};
    kwargs...,
) where {𝔽,M,O<:AbstractGroupOperation}
    return nothing
end

function check_point(
    G::AbstractGroupManifold{𝔽,O1},
    e::Identity{O2};
    kwargs...,
) where {𝔽,M,O1<:AbstractGroupOperation,O2<:AbstractGroupOperation}
    return DomainError(
        e,
        "The Identity $e does not lie on $G, since its the identity with respect to $O2 and not $O1.",
    )
end

##########################
# Group-specific functions
##########################

@doc raw"""
    adjoint_action(G::AbstractGroupManifold, p, X)

Adjoint action of the element `p` of the Lie group `G` on the element `X`
of the corresponding Lie algebra.

It is defined as the differential of the group authomorphism ``Ψ_p(q) = pqp⁻¹`` at
the identity of `G`.

The formula reads
````math
\operatorname{Ad}_p(X) = dΨ_p(e)[X]
````
where $e$ is the identity element of `G`.

Note that the adjoint representation of a Lie group isn't generally faithful.
Notably the adjoint representation of SO(2) is trivial.
"""
adjoint_action(G::AbstractGroupManifold, p, X)
@decorator_transparent_function :intransparent function adjoint_action(
    G::AbstractGroupManifold,
    p,
    Xₑ,
)
    Xₚ = translate_diff(G, p, Identity(G), Xₑ, LeftAction())
    Y = inverse_translate_diff(G, p, p, Xₚ, RightAction())
    return Y
end

function adjoint_action!(G::AbstractGroupManifold, Y, p, Xₑ)
    Xₚ = translate_diff(G, p, Identity(G), Xₑ, LeftAction())
    inverse_translate_diff!(G, Y, p, p, Xₚ, RightAction())
    return Y
end

@doc raw"""
    inv(G::AbstractGroupManifold, p)

Inverse $p^{-1} ∈ \mathcal{G}$ of an element $p ∈ \mathcal{G}$, such that
$p \circ p^{-1} = p^{-1} \circ p = e ∈ \mathcal{G}$, where $e$ is the [`Identity`](@ref)
element of $\mathcal{G}$.
"""
inv(::AbstractGroupManifold, ::Any...)
@decorator_transparent_function function Base.inv(G::AbstractGroupManifold, p)
    q = allocate_result(G, inv, p)
    return inv!(G, q, p)
end

function Base.inv(
    ::AbstractGroupManifold{𝔽,O},
    e::Identity{O},
) where {𝔽,O<:AbstractGroupOperation}
    return e
end

@decorator_transparent_function function inv!(G::AbstractGroupManifold, q, p)
    return inv!(G.manifold, q, p)
end

function inv!(
    G::AbstractGroupManifold{𝔽,O},
    q,
    ::Identity{O},
) where {𝔽,O<:AbstractGroupOperation}
    return identity_element!(G, q)
end

function Base.copyto!(
    ::AbstractGroupManifold{𝔽,O},
    e::Identity{O},
    ::Identity{O},
) where {𝔽,O<:AbstractGroupOperation}
    return e
end
function Base.copyto!(
    G::AbstractGroupManifold{𝔽,O},
    p,
    ::Identity{O},
) where {𝔽,O<:AbstractGroupOperation}
    return identity_element!(G, p)
end

@doc raw"""
    compose(G::AbstractGroupManifold, p, q)

Compose elements ``p,q ∈ \mathcal{G}`` using the group operation ``p \circ q``.

For implementing composition on a new group manifold, please overload `_compose`
instead so that methods with [`Identity`](@ref) arguments are not ambiguous.
"""
compose(::AbstractGroupManifold, ::Any...)

@decorator_transparent_function function compose(
    G::AbstractGroupManifold{𝔽,Op},
    p,
    q,
) where {𝔽,Op<:AbstractGroupOperation}
    return _compose(G, p, q)
end
function compose(
    ::AbstractGroupManifold{𝔽,Op},
    ::Identity{Op},
    p,
) where {𝔽,Op<:AbstractGroupOperation}
    return p
end
function compose(
    ::AbstractGroupManifold{𝔽,Op},
    p,
    ::Identity{Op},
) where {𝔽,Op<:AbstractGroupOperation}
    return p
end
function compose(
    ::AbstractGroupManifold{𝔽,Op},
    e::Identity{Op},
    ::Identity{Op},
) where {𝔽,Op<:AbstractGroupOperation}
    return e
end

function _compose(G::AbstractGroupManifold, p, q)
    x = allocate_result(G, compose, p, q)
    return _compose!(G, x, p, q)
end

@decorator_transparent_signature compose!(M::AbstractDecoratorManifold, x, p, q)

compose!(G::AbstractGroupManifold, x, q, p) = _compose!(G, x, q, p)
function compose!(
    G::AbstractGroupManifold{𝔽,Op},
    q,
    p,
    ::Identity{Op},
) where {𝔽,Op<:AbstractGroupOperation}
    return copyto!(G, q, p)
end
function compose!(
    G::AbstractGroupManifold{𝔽,Op},
    q,
    ::Identity{Op},
    p,
) where {𝔽,Op<:AbstractGroupOperation}
    return copyto!(G, q, p)
end
function compose!(
    G::AbstractGroupManifold{𝔽,Op},
    q,
    ::Identity{Op},
    e::Identity{Op},
) where {𝔽,Op<:AbstractGroupOperation}
    return identity_element!(G, q)
end
function compose!(
    ::AbstractGroupManifold{𝔽,Op},
    e::Identity{Op},
    ::Identity{Op},
    ::Identity{Op},
) where {𝔽,Op<:AbstractGroupOperation}
    return e
end

transpose(e::Identity) = e

@doc raw"""
    hat(M::AbstractGroupManifold{𝔽,O}, ::Identity{O}, Xⁱ) where {𝔽,O<:AbstractGroupOperation}

Given a basis $e_i$ on the tangent space at a the [`Identity`}(@ref) and tangent
component vector ``X^i``, compute the equivalent vector representation
``X=X^i e_i**, where Einstein summation notation is used:

````math
∧ : X^i ↦ X^i e_i
````

For array manifolds, this converts a vector representation of the tangent
vector to an array representation. The [`vee`](@ref) map is the `hat` map's
inverse.
"""
function hat(
    G::AbstractGroupManifold{𝔽,O},
    ::Identity{O},
    X,
) where {𝔽,O<:AbstractGroupOperation}
    return get_vector_lie(G, X, VeeOrthogonalBasis())
end
function hat!(
    G::AbstractGroupManifold{𝔽,O},
    Y,
    ::Identity{O},
    X,
) where {𝔽,O<:AbstractGroupOperation}
    return get_vector_lie!(G, Y, X, VeeOrthogonalBasis())
end
function hat(M::AbstractManifold, e::Identity, ::Any)
    return throw(ErrorException("On $M there exsists no identity $e"))
end
function hat!(M::AbstractManifold, ::Any, e::Identity, ::Any)
    return throw(ErrorException("On $M there exsists no identity $e"))
end

@doc raw"""
    vee(M::AbstractManifold, p, X)

Given a basis $e_i$ on the tangent space at a point `p` and tangent
vector `X`, compute the vector components $X^i$, such that $X = X^i e_i$, where
Einstein summation notation is used:

````math
\vee : X^i e_i ↦ X^i
````

For array manifolds, this converts an array representation of the tangent
vector to a vector representation. The [`hat`](@ref) map is the `vee` map's
inverse.
"""
function vee(
    M::AbstractGroupManifold{𝔽,O},
    ::Identity{O},
    X,
) where {𝔽,O<:AbstractGroupOperation}
    return get_coordinates_lie(M, X, VeeOrthogonalBasis())
end
function vee!(
    M::AbstractGroupManifold{𝔽,O},
    Y,
    ::Identity{O},
    X,
) where {𝔽,O<:AbstractGroupOperation}
    return get_coordinates_lie!(M, Y, X, VeeOrthogonalBasis())
end
function vee(M::AbstractManifold, e::Identity, ::Any)
    return throw(ErrorException("On $M there exsists no identity $e"))
end
function vee!(M::AbstractManifold, ::Any, e::Identity, ::Any)
    return throw(ErrorException("On $M there exsists no identity $e"))
end

"""
    lie_bracket(G::AbstractGroupManifold, X, Y)

Lie bracket between elements `X` and `Y` of the Lie algebra corresponding to the Lie group `G`.

This can be used to compute the adjoint representation of a Lie algebra.
Note that this representation isn't generally faithful. Notably the adjoint
representation of 𝔰𝔬(2) is trivial.
"""
lie_bracket(G::AbstractGroupManifold, X, Y)
@decorator_transparent_signature lie_bracket(M::AbstractDecoratorManifold, X, Y)

_action_order(p, q, ::LeftAction) = (p, q)
_action_order(p, q, ::RightAction) = (q, p)

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
@decorator_transparent_function function translate(G::AbstractGroupManifold, p, q)
    return translate(G, p, q, LeftAction())
end
@decorator_transparent_function function translate(
    G::AbstractGroupManifold,
    p,
    q,
    conv::ActionDirection,
)
    return compose(G, _action_order(p, q, conv)...)
end

@decorator_transparent_function function translate!(G::AbstractGroupManifold, X, p, q)
    return translate!(G, X, p, q, LeftAction())
end
@decorator_transparent_function function translate!(
    G::AbstractGroupManifold,
    X,
    p,
    q,
    conv::ActionDirection,
)
    return compose!(G, X, _action_order(p, q, conv)...)
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
@decorator_transparent_function function inverse_translate(G::AbstractGroupManifold, p, q)
    return inverse_translate(G, p, q, LeftAction())
end
@decorator_transparent_function function inverse_translate(
    G::AbstractGroupManifold,
    p,
    q,
    conv::ActionDirection,
)
    return translate(G, inv(G, p), q, conv)
end

@decorator_transparent_function function inverse_translate!(
    G::AbstractGroupManifold,
    X,
    p,
    q,
)
    return inverse_translate!(G, X, p, q, LeftAction())
end
@decorator_transparent_function function inverse_translate!(
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
@decorator_transparent_function function translate_diff(G::AbstractGroupManifold, p, q, X)
    return translate_diff(G, p, q, X, LeftAction())
end
@decorator_transparent_function function translate_diff(
    G::AbstractGroupManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
    Y = allocate_result(G, translate_diff, X, p, q)
    translate_diff!(G, Y, p, q, X, conv)
    return Y
end

@decorator_transparent_function function translate_diff!(
    G::AbstractGroupManifold,
    Y,
    p,
    q,
    X,
)
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
@decorator_transparent_function function inverse_translate_diff(
    G::AbstractGroupManifold,
    p,
    q,
    X,
)
    return inverse_translate_diff(G, p, q, X, LeftAction())
end
@decorator_transparent_function function inverse_translate_diff(
    G::AbstractGroupManifold,
    p,
    q,
    X,
    conv::ActionDirection,
)
    return translate_diff(G, inv(G, p), q, X, conv)
end

@decorator_transparent_function function inverse_translate_diff!(
    G::AbstractGroupManifold,
    Y,
    p,
    q,
    X,
)
    return inverse_translate_diff!(G, Y, p, q, X, LeftAction())
end
@decorator_transparent_function function inverse_translate_diff!(
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
    exp_lie(G::AbstractGroupManifold, X)

Compute the group exponential of the Lie algebra element `X`. It is equivalent to the
exponential map defined by the [`CartanSchoutenMinus`](@ref) connection.

Given an element $X ∈ 𝔤 = T_e \mathcal{G}$, where $e$ is the [`Identity`](@ref) element of
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
exp_lie(G::AbstractGroupManifold{𝔽,AdditionOperation}, X) where {𝔽}
```

Compute $q = X$.

    exp_lie(G::AbstractGroupManifold{𝔽,MultiplicationOperation}, X) where {𝔽}

For `Number` and `AbstractMatrix` types of `X`, compute the usual numeric/matrix
exponential,

````math
\exp X = \operatorname{Exp} X = \sum_{n=0}^∞ \frac{1}{n!} X^n.
````
"""
exp_lie(::AbstractGroupManifold, ::Any...)
@decorator_transparent_function function exp_lie(G::AbstractGroupManifold, X)
    q = allocate_result(G, exp_lie, X)
    return exp_lie!(G, q, X)
end

@decorator_transparent_signature exp_lie!(M::AbstractDecoratorManifold, q, X)

@doc raw"""
    log_lie(G::AbstractGroupManifold, q)
    log_lie!(G::AbstractGroupManifold, X, q)

Compute the Lie group logarithm of the Lie group element `q`. It is equivalent to the
logarithmic map defined by the [`CartanSchoutenMinus`](@ref) connection.

Given an element $q ∈ \mathcal{G}$, compute the right inverse of the group exponential map
[`exp_lie`](@ref), that is, the element $\log q = X ∈ 𝔤 = T_e \mathcal{G}$, such that
$q = \exp X$

!!! note
    In general, the group logarithm map is distinct from the Riemannian logarithm map
    [`log`](@ref).

    For matrix Llie groups this is equal to the (matrix) logarithm:

````math
\log q = \operatorname{Log} q = \sum_{n=1}^∞ \frac{(-1)^{n+1}}{n} (q - e)^n,
````

where $e$ here is the [`Identity`](@ref) element, that is, $1$ for numeric $q$ or the
identity matrix $I_m$ for matrix $q ∈ ℝ^{m × m}$.

Since this function handles [`Identity`](@ref) arguments, the preferred function to override
is `_log_lie!`.
"""
log_lie(::AbstractGroupManifold, ::Any...)
@decorator_transparent_function function log_lie(G::AbstractGroupManifold, q)
    X = allocate_result(G, log_lie, q)
    return log_lie!(G, X, q)
end
function log_lie(
    G::AbstractGroupManifold{𝔽,Op},
    ::Identity{Op},
) where {𝔽,Op<:AbstractGroupOperation}
    return zero_vector(G, identity_element(G))
end

@decorator_transparent_function function log_lie!(G::AbstractGroupManifold, X, q)
    return _log_lie!(G, X, q)
end

function log_lie!(
    G::AbstractGroupManifold{𝔽,Op},
    X,
    ::Identity{Op},
) where {𝔽,Op<:AbstractGroupOperation}
    return zero_vector!(G, X, identity_element(G))
end

############################
# Group-specific Retractions
############################

"""
    GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod

Retraction using the group exponential [`exp_lie`](@ref) "translated" to any point on the
manifold.

For more details, see
[`retract`](@ref retract(::GroupManifold, p, X, ::GroupExponentialRetraction)).

# Constructor

    GroupExponentialRetraction(conv::ActionDirection = LeftAction())
"""
struct GroupExponentialRetraction{D<:ActionDirection} <: AbstractRetractionMethod end

function GroupExponentialRetraction(conv::ActionDirection=LeftAction())
    return GroupExponentialRetraction{typeof(conv)}()
end

"""
    GroupLogarithmicInverseRetraction{D<:ActionDirection} <: AbstractInverseRetractionMethod

Retraction using the group logarithm [`log_lie`](@ref) "translated" to any point on the
manifold.

For more details, see
[`inverse_retract`](@ref inverse_retract(::GroupManifold, p, q ::GroupLogarithmicInverseRetraction)).

# Constructor

    GroupLogarithmicInverseRetraction(conv::ActionDirection = LeftAction())
"""
struct GroupLogarithmicInverseRetraction{D<:ActionDirection} <:
       AbstractInverseRetractionMethod end

function GroupLogarithmicInverseRetraction(conv::ActionDirection=LeftAction())
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

Compute the retraction using the group exponential [`exp_lie`](@ref) "translated" to any
point on the manifold.
With a group translation ([`translate`](@ref)) $τ_p$ in a specified direction, the
retraction is

````math
\operatorname{retr}_p = τ_p \circ \exp \circ (\mathrm{d}τ_p^{-1})_p,
````

where $\exp$ is the group exponential ([`exp_lie`](@ref)), and $(\mathrm{d}τ_p^{-1})_p$ is
the action of the differential of inverse translation $τ_p^{-1}$ evaluated at $p$ (see
[`inverse_translate_diff`](@ref)).
"""
function retract(G::AbstractGroupManifold, p, X, method::GroupExponentialRetraction)
    conv = direction(method)
    Xₑ = inverse_translate_diff(G, p, p, X, conv)
    pinvq = exp_lie(G, Xₑ)
    q = translate(G, p, pinvq, conv)
    return q
end

function retract!(G::AbstractGroupManifold, q, p, X, method::GroupExponentialRetraction)
    conv = direction(method)
    Xₑ = inverse_translate_diff(G, p, p, X, conv)
    pinvq = exp_lie(G, Xₑ)
    return translate!(G, q, p, pinvq, conv)
end

@doc raw"""
    inverse_retract(
        G::AbstractGroupManifold,
        p,
        X,
        method::GroupLogarithmicInverseRetraction{<:ActionDirection},
    )

Compute the inverse retraction using the group logarithm [`log_lie`](@ref) "translated"
to any point on the manifold.
With a group translation ([`translate`](@ref)) $τ_p$ in a specified direction, the
retraction is

````math
\operatorname{retr}_p^{-1} = (\mathrm{d}τ_p)_e \circ \log \circ τ_p^{-1},
````

where $\log$ is the group logarithm ([`log_lie`](@ref)), and $(\mathrm{d}τ_p)_e$ is the
action of the differential of translation $τ_p$ evaluated at the identity element $e$
(see [`translate_diff`](@ref)).
"""
function inverse_retract(G::GroupManifold, p, q, method::GroupLogarithmicInverseRetraction)
    conv = direction(method)
    pinvq = inverse_translate(G, p, q, conv)
    Xₑ = log_lie(G, pinvq)
    return translate_diff(G, p, Identity(G), Xₑ, conv)
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
    Xₑ = log_lie(G, pinvq)
    return translate_diff!(G, X, p, Identity(G), Xₑ, conv)
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

Base.:+(e::Identity{AdditionOperation}) = e
Base.:+(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e
Base.:+(::Identity{AdditionOperation}, p) = p
Base.:+(p, ::Identity{AdditionOperation}) = p

Base.:-(e::Identity{AdditionOperation}) = e
Base.:-(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e
Base.:-(::Identity{AdditionOperation}, p) = -p
Base.:-(p, ::Identity{AdditionOperation}) = p

Base.:*(e::Identity{AdditionOperation}, p) = e
Base.:*(p, e::Identity{AdditionOperation}) = e
Base.:*(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e

adjoint_action(::AdditionGroup, p, X) = X

adjoint_action!(G::AdditionGroup, Y, p, X) = copyto!(G, Y, p, X)

identity_element(::AdditionGroup, p::Number) = zero(p)

function identity_element!(::AbstractGroupManifold{𝔽,<:AdditionOperation}, p) where {𝔽}
    return fill!(p, zero(eltype(p)))
end

Base.inv(::AdditionGroup, p) = -p
Base.inv(::AdditionGroup, e::Identity) = e

inv!(G::AdditionGroup, q, p) = copyto!(G, q, -p)
inv!(G::AdditionGroup, q, ::Identity{AdditionOperation}) = identity_element!(G, q)
inv!(::AdditionGroup, q::Identity{AdditionOperation}, e::Identity{AdditionOperation}) = q

function is_identity(G::AdditionGroup, q; kwargs...)
    return isapprox(G, q, zero(q); kwargs...)
end
function is_identity(G::AdditionGroup, e::Identity; kwargs...)
    return invoke(is_identity, Tuple{AbstractGroupManifold,typeof(e)}, G, e; kwargs...)
end

_compose(::AdditionGroup, p, q) = p + q

function _compose!(::AdditionGroup, x, p, q)
    x .= p .+ q
    return x
end

translate_diff(::AdditionGroup, p, q, X, ::ActionDirection) = X

translate_diff!(G::AdditionGroup, Y, p, q, X, ::ActionDirection) = copyto!(G, Y, p, X)

inverse_translate_diff(::AdditionGroup, p, q, X, ::ActionDirection) = X

function inverse_translate_diff!(G::AdditionGroup, Y, p, q, X, ::ActionDirection)
    return copyto!(G, Y, p, X)
end

exp_lie(::AdditionGroup, X) = X

exp_lie!(G::AdditionGroup, q, X) = copyto!(G, q, X)

log_lie(::AdditionGroup, q) = q
function log_lie(G::AdditionGroup, ::Identity{AdditionOperation})
    return zero_vector(G, identity_element(G))
end

_log_lie!(G::AdditionGroup, X, q) = copyto!(G, X, q)

lie_bracket(::AdditionGroup, X, Y) = zero(X)

lie_bracket!(::AdditionGroup, Z, X, Y) = fill!(Z, 0)

#######################################
# Overloads for MultiplicationOperation
#######################################

"""
    MultiplicationOperation <: AbstractGroupOperation

Group operation that consists of multiplication.
"""
struct MultiplicationOperation <: AbstractGroupOperation end

const MultiplicationGroup = AbstractGroupManifold{𝔽,MultiplicationOperation} where {𝔽}

Base.:*(e::Identity{MultiplicationOperation}) = e
Base.:*(::Identity{MultiplicationOperation}, p) = p
Base.:*(p, ::Identity{MultiplicationOperation}) = p
Base.:*(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e
Base.:*(::Identity{MultiplicationOperation}, e::Identity{AdditionOperation}) = e
Base.:*(e::Identity{AdditionOperation}, ::Identity{MultiplicationOperation}) = e

Base.:/(p, ::Identity{MultiplicationOperation}) = p
Base.:/(::Identity{MultiplicationOperation}, p) = inv(p)
Base.:/(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e

Base.:\(p, ::Identity{MultiplicationOperation}) = inv(p)
Base.:\(::Identity{MultiplicationOperation}, p) = p
Base.:\(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e

LinearAlgebra.det(::Identity{MultiplicationOperation}) = true
LinearAlgebra.adjoint(e::Identity{MultiplicationOperation}) = e

function identity_element!(::MultiplicationGroup, p::AbstractMatrix)
    return copyto!(p, I)
end

function identity_element!(G::MultiplicationGroup, p::AbstractArray)
    if length(p) == 1
        fill!(p, one(eltype(p)))
    else
        throw(DimensionMismatch("Array $p cannot be set to identity element of group $G"))
    end
    return p
end

function is_identity(G::MultiplicationGroup, q::Number; kwargs...)
    return isapprox(G, q, one(q); kwargs...)
end
function is_identity(G::MultiplicationGroup, q::AbstractVector; kwargs...)
    return length(q) == 1 && isapprox(G, q[], one(q[]); kwargs...)
end
function is_identity(G::MultiplicationGroup, q::AbstractMatrix; kwargs...)
    return isapprox(G, q, I; kwargs...)
end
function is_identity(G::MultiplicationGroup, e::Identity; kwargs...)
    return invoke(is_identity, Tuple{AbstractGroupManifold,typeof(e)}, G, e; kwargs...)
end

LinearAlgebra.mul!(q, ::Identity{MultiplicationOperation}, p) = copyto!(q, p)
LinearAlgebra.mul!(q, p, ::Identity{MultiplicationOperation}) = copyto!(q, p)
function LinearAlgebra.mul!(
    q::AbstractMatrix,
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return copyto!(q, I)
end
function LinearAlgebra.mul!(
    q,
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return copyto!(q, one(q))
end
function LinearAlgebra.mul!(
    q::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return q
end
Base.one(e::Identity{MultiplicationOperation}) = e

Base.inv(::MultiplicationGroup, p) = inv(p)
Base.inv(::MultiplicationGroup, e::Identity{MultiplicationOperation}) = e

inv!(G::MultiplicationGroup, q, p) = copyto!(q, inv(G, p))
function inv!(G::MultiplicationGroup, q, ::Identity{MultiplicationOperation})
    return identity_element!(G, q)
end
function inv!(
    ::MultiplicationGroup,
    q::Identity{MultiplicationOperation},
    e::Identity{MultiplicationOperation},
)
    return q
end

_compose(::MultiplicationGroup, p, q) = p * q

_compose!(::MultiplicationGroup, x, p, q) = mul!_safe(x, p, q)

inverse_translate(::MultiplicationGroup, p, q, ::LeftAction) = p \ q
inverse_translate(::MultiplicationGroup, p, q, ::RightAction) = q / p

function inverse_translate!(G::MultiplicationGroup, x, p, q, conv::ActionDirection)
    return copyto!(x, inverse_translate(G, p, q, conv))
end

function exp_lie!(G::MultiplicationGroup, q, X)
    X isa Union{Number,AbstractMatrix} && return copyto!(q, exp(X))
    return error(
        "exp_lie! not implemented on $(typeof(G)) for vector $(typeof(X)) and element $(typeof(q)).",
    )
end

log_lie!(::MultiplicationGroup, X::AbstractMatrix, q::AbstractMatrix) = log_safe!(X, q)

lie_bracket(::MultiplicationGroup, X, Y) = mul!(X * Y, Y, X, -1, true)

function lie_bracket!(::MultiplicationGroup, Z, X, Y)
    mul!(Z, X, Y)
    mul!(Z, Y, X, -1, true)
    return Z
end

@doc raw"""
    get_vector_lie(G::AbstractGroupManifold, a, B::AbstractBasis)

Reconstruct a tangent vector from the Lie algebra of `G` from cooordinates `a` of a basis `B`.
This is similar to calling [`get_vector`](@ref) at the `p=`[`Identity`](@ref)`(G)`.
"""
function get_vector_lie(G::AbstractGroupManifold, X, B::AbstractBasis)
    return get_vector(G, identity_element(G), X, B)
end
function get_vector_lie!(G::AbstractGroupManifold, Y, X, B::AbstractBasis)
    return get_vector!(G, Y, identity_element(G), X, B)
end

@doc raw"""
    get_coordinates_lie(G::AbstractGroupManifold, X, B::AbstractBasis)

Get the coordinates of an element `X` from the Lie algebra og `G` with respect to a basis `B`.
This is similar to calling [`get_coordinates`](@ref) at the `p=`[`Identity`](@ref)`(G)`.
"""
function get_coordinates_lie(G::AbstractGroupManifold, X, B::AbstractBasis)
    return get_coordinates(G, identity_element(G), X, B)
end
function get_coordinates_lie!(G::AbstractGroupManifold, a, X, B::AbstractBasis)
    return get_coordinates!(G, a, identity_element(G), X, B)
end

# (a) changes / parent.
for f in [
    embed,
    get_basis,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
]
    eval(
        quote
            function decorator_transparent_dispatch(
                ::typeof($f),
                ::AbstractGroupManifold{𝔽,O,<:AbstractGroupDecoratorType},
                args...,
            ) where {𝔽,O}
                return Val(:parent)
            end
        end,
    )
end
for f in [
    get_coordinates,
    get_coordinates!,
    get_vector,
    get_vector!,
    inverse_retract!,
    mid_point!,
    project,
    retract!,
    vector_transport_along,
]
    eval(
        quote
            function decorator_transparent_dispatch(
                ::typeof($f),
                ::AbstractGroupManifold{𝔽,O,DefaultGroupDecoratorType},
                args...,
            ) where {𝔽,O}
                return Val(:parent)
            end
        end,
    )
end
# (b) changes / transparencies
for f in [
    check_point,
    check_vector,
    copy,
    copyto!,
    distance,
    exp,
    exp!,
    embed!,
    get_coordinates,
    get_coordinates!,
    get_vector,
    get_vector!,
    inner,
    inverse_retract,
    inverse_retract!,
    isapprox,
    log,
    log!,
    mid_point,
    mid_point!,
    project!,
    project,
    retract,
    retract!,
    vector_transport_along,
    vector_transport_direction,
]
    eval(
        quote
            function decorator_transparent_dispatch(
                ::typeof($f),
                ::AbstractGroupManifold{𝔽,O,<:TransparentGroupDecoratorType},
                args...,
            ) where {𝔽,O}
                return Val(:transparent)
            end
        end,
    )
end

# (c) changes / intransparencies.
for f in [
    compose,
    compose!,
    exp_lie,
    exp_lie!,
    log_lie,
    log_lie!,
    translate,
    translate!,
    translate_diff,
    translate_diff!,
]
    eval(
        quote
            function decorator_transparent_dispatch(
                ::typeof($f),
                ::AbstractGroupManifold,
                args...,
            )
                return Val(:intransparent)
            end
        end,
    )
end
# (d) specials
for f in [vector_transport_along!, vector_transport_direction!, vector_transport_to!]
    eval(
        quote
            function decorator_transparent_dispatch(
                ::typeof($f),
                ::AbstractGroupManifold{𝔽,O,<:TransparentGroupDecoratorType},
                Y,
                p,
                X,
                q,
                ::T,
            ) where {𝔽,O,T}
                return Val(:transparent)
            end
            function decorator_transparent_dispatch(
                ::typeof($f),
                ::AbstractGroupManifold{𝔽,O,<:AbstractGroupDecoratorType},
                Y,
                p,
                X,
                q,
                ::T,
            ) where {𝔽,O,T}
                return Val(:intransparent)
            end
        end,
    )
    for m in [PoleLadderTransport, SchildsLadderTransport, ScaledVectorTransport]
        eval(
            quote
                function decorator_transparent_dispatch(
                    ::typeof($f),
                    ::AbstractGroupManifold{𝔽,O,<:TransparentGroupDecoratorType},
                    Y,
                    p,
                    X,
                    q,
                    ::$m,
                ) where {𝔽,O}
                    return Val(:parent)
                end
                function decorator_transparent_dispatch(
                    ::typeof($f),
                    ::AbstractGroupManifold{𝔽,O,<:AbstractGroupDecoratorType},
                    Y,
                    p,
                    X,
                    q,
                    ::$m,
                ) where {𝔽,O}
                    return Val(:parent)
                end
            end,
        )
    end
end
