"""
    AbstractGroupOperation

Abstract type for smooth binary operations on elements of a Lie group.
"""
abstract type AbstractGroupOperation end

@doc doc"""
    AbstractGroupManifold{<:AbstractGroupOperation} <: Manifold

Abstract type for a Lie group (a group that is also a smooth manifold with a
smooth binary operation). `AbstractGroupManifold`s must implement at least
`inv`(@ref), `identity`(@ref), and `left_action`(@ref).

Group manifolds by default assume a [`LeftInvariantCanonicalMetric`](@ref)
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
    inv(G::AbstractGroupManifold, x)

Inverse $x^{-1}$ of an element $x$, such that
$x \cdot x^{-1} = x^{-1} \cdot x = e$.
"""
function inv(G::AbstractGroupManifold, x)
    error("inv not implemented on $(typeof(G)) for point $(typeof(x))")
end

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

"""
    compose(G::AbstractGroupManifold, x...)

Compose elements of `G` using their left translation upon each other.
"""
function compose(G::AbstractGroupManifold, x...)
    return reduce((x, y)->left_action(G, x, y), x)
end

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
function exp!(G::AbstractGroupManifold, y, v)
    error("Exponential map not implemented on $(typeof(G)) for point $(typeof(y)) and tangent vector $(typeof(v))")
end

"""
    exp!(G::AbstractGroupManifold, y, v, t::Real)

Exponential map of tangent vector `t*v` at the identity element of group `G`.
Result is saved to `y`.
"""
exp!(G::AbstractGroupManifold, y, v, t::Real) = exp!(G, y, t*v)

function exp!(G::AbstractGroupManifold, y, x, v)
    exp!(G, y, \(G, x, v))
    y .= compose(G, x, y)
    return y
end

"""
    exp(G::AbstractGroupManifold, v)

Exponential map of tangent vector `v` at the identity element of group `G`.
"""
function exp(G::AbstractGroupManifold, v)
    y = similar(v)
    exp!(G, y, v)
    return y
end

"""
    exp(G::AbstractGroupManifold, v, t::Real)

Exponential map of tangent vector `t*v` at the identity element of group `G`.
"""
exp(G::AbstractGroupManifold, v, t::Real) = exp(G, t*v)

function log!(G::AbstractGroupManifold, v, y)
    error("Logarithmic map not implemented on $(typeof(G)) for point $(typeof(y)) and vector $(typeof(v))")
end

function log!(G::AbstractGroupManifold, v, x, y)
    log!(G, v, \(G, x, y))
    v .= left_action(G, x, v)
end

function log(G::AbstractGroupManifold, y)
    v = similar_result(G, log, y)
    log!(G, v, y)
    return v
end


"""
    AdditionOperation <: AbstractGroupOperation

Group operation that consists of simple addition.
"""
struct AdditionOperation <: AbstractGroupOperation end

inv(::AbstractGroupManifold{AdditionOperation}, x) = -x

identity(::AbstractGroupManifold{AdditionOperation}, x) = zero(x)

left_action(::AbstractGroupManifold{AdditionOperation}, x, p) = x + p

@doc doc"""
    right_action(::AbstractGroupManifold{AdditionOperation}, x, p)

For an element $p$ of some set, compute the right action of group element $x$
on $p$, i.e. $p \cdot x$. For an `AdditionOperation`, the right and left
actions are the same.
"""
function right_action(G::AbstractGroupManifold{AdditionOperation}, x, p)
    return left_action(G, x, p)
end

compose(::AbstractGroupManifold{AdditionOperation}, x...) = +(x...)

lie_bracket(::AbstractGroupManifold{AdditionOperation}, v, w) = zero(v)

exp!(::AbstractGroupManifold{AdditionOperation}, y, v) = y .= v

log!(::AbstractGroupManifold{AdditionOperation}, v, y) = v .= y


"""
    MultiplicationOperation <: AbstractGroupOperation

Group operation that consists of multiplication.
"""
struct MultiplicationOperation <: AbstractGroupOperation end

inv(::AbstractGroupManifold{MultiplicationOperation}, x) = inv(x)

identity(::AbstractGroupManifold{MultiplicationOperation}, x) = one(x)

left_action(::AbstractGroupManifold{MultiplicationOperation}, x, p) = x * p

compose(::AbstractGroupManifold{MultiplicationOperation}, x...) = *(x...)

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

@traitfn function exp!(G::GT, y, v) where {O<:MultiplicationOperation,
                                           GT<:AbstractGroupManifold{O};
                                           !IsMatrixGroup{GT}}
    error("Exponential map not implemented on $(typeof(G)) for point $(typeof(y)) and tangent vector $(typeof(v))")
end

@traitfn function exp!(G::GT, y, v) where {O<:MultiplicationOperation,
                                           GT<:AbstractGroupManifold{O};
                                           IsMatrixGroup{GT}}
    y .= exp(v)
    return y
end

_log(x) = log(x)
_log(x::StaticArray) = log(Matrix(x))

@traitfn function log!(G::GT, v, y) where {O<:MultiplicationOperation,
                                           GT<:AbstractGroupManifold{O};
                                           !IsMatrixGroup{GT}}
    error("Logarithmic map not implemented on $(typeof(G)) for point $(typeof(y)) and vector $(typeof(v))")
end

@traitfn function log!(G::GT, v, y) where {O<:MultiplicationOperation,
                                           GT<:AbstractGroupManifold{O};
                                           IsMatrixGroup{GT}}
    w = _log(y)
    v .= isreal(v) ? real(w) : w
    return v
end


"""
    QuaternionMultiplicationOperation <: AbstractGroupOperation

Group operation that consists of quaternion multiplication, also known as
the Hamilton product.
"""
struct QuaternionMultiplicationOperation <: AbstractGroupOperation end

function hamilton_prod!(z, x, y)
    @assert length(x) == 4 && length(y) == 4 && length(z) == 4

    @inbounds begin
        xs = x[1]
        xv1 = x[2]
        xv2 = x[3]
        xv3 = x[4]
        ys = y[1]
        yv1 = y[2]
        yv2 = y[3]
        yv3 = y[4]

        z[1] = xs * ys - xv1 * yv1 - xv2 * yv2 - xv3 * yv3
        z[2] = xs * yv1 + xv1 * ys + xv2 * yv3 - xv3 * yv2
        z[3] = xs * yv2 - xv1 * yv3 + xv2 * ys + xv3 * yv1
        z[4] = xs * yv3 + xv1 * yv2 - xv2 * yv1 + xv3 * ys
    end

    return z
end

function hamilton_prod(x, y)
    z = similar(x)
    hamilton_prod!(z, x, y)
    return z
end

function inv(::AbstractGroupManifold{QuaternionMultiplicationOperation}, x)
    @assert length(x) == 4
    y = similar(x)
    vi = SVector{3}(2:4)
    @inbounds begin
        y[1] = x[1]
        y[vi] .= -x[vi]
    end
    return y
end

function identity(::AbstractGroupManifold{QuaternionMultiplicationOperation}, x)
    e = similar(x, 4)
    vi = SVector{3}(2:4)
    @inbounds begin
        e[1] = 1
        e[vi] .= 0
    end
    return e
end

function left_action(::AbstractGroupManifold{QuaternionMultiplicationOperation},
                     x,
                     p)
    return hamilton_prod(x, p)
end

function exp!(::AbstractGroupManifold{QuaternionMultiplicationOperation}, y, v)
      @assert length(v) == 4 && length(y) == 4
      vi = SVector{3}(2:4)
      @inbounds begin
          θu = v[vi]
          θ = norm(θu)
          y[1] = cos(θ)
          y[vi] .= (θ ≈ 0 ? 1 - θ^2 / 6 : sin(θ) / θ) .* θu
      end
      return y
end

function log!(::AbstractGroupManifold{QuaternionMultiplicationOperation}, v, y)
    @assert length(v) == 4 && length(y) == 4
    vi = SVector{3}(2:4)
    @inbounds begin
        sinθv = view(y, vi)
        θ = atan(norm(sinθv), y[1])
        v[1] = 0
        v[vi] .= (θ ≈ 0 ? 1 + θ^2 / 6 : θ / sin(θ)) .* sinθv
    end
    return v
end
