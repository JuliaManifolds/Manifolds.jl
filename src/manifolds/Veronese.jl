@doc raw"""
    Veronese{T} <: AbstractManifold{ℝ}

The Veronese manifold

````math
\mathcal V_{N,D}
=
\left\{
    λ x^{⊗ D}
    \;\middle|\;
    λ\in\mathbb R\setminus\{0\},\;
    x\in\mathbb S^{N-1}
\right\}
````

is the set of nonzero symmetric rank-one tensors of order ``D`` over ``ℝ^N``,
where ``N`` and ``D`` are positive integers. Here

````math
x^{⊗ D}
:=
\underbrace{x⊗\cdots⊗x}_{D\text{ factors}}
````

is the ``D``-fold tensor product of ``x`` with itself. The parametrization

````math
Φ:(\mathbb R\setminus\{0\})\times\mathbb S^{N-1}\to\mathcal V_{N,D},
\qquad
Φ(λ,x)=λ x^{⊗ D},
````

is two-to-one. Every embedded tensor can be parametrized by either of the two
pairs

````math
(λ,x)
\sim
\bigl((-1)^Dλ,-x\bigr).
````

Hence, for even ``D`` the sign of ``λ`` is intrinsic and
``\mathcal V_{N,D}`` has two connected components, while for odd ``D`` the sign
can be absorbed by replacing ``x`` by ``-x`` and the manifold is connected
when ``N≥2``. For ``N=1``, the manifold is ``ℝ\setminus\{0\}`` for every
``D`` and has two connected components.

An instance `M = Veronese(N, D)` represents the manifold
``\mathcal V_{N,D}``. In the implementation, a manifold point is stored by
choosing one of the parameter pairs above. Thus

````math
M=\mathcal V_{N,D},
\qquad
p\leftrightarrow(λ,x),
\qquad
Φ(λ,x)=λ x^{⊗ D}.
````

Concretely, the chosen pair is stored as `p = ([λ], x)`. A tangent vector is
stored analogously as `X = ([ν], u)`, where

````math
u\in T_x\mathbb S^{N-1}=x^⊥.
````

The metric is induced by the Euclidean metric on the full tensor space
``(ℝ^N)^{⊗ D}``. The differential of ``Φ`` is

````math
DΦ_{(λ,x)}(ν,u)
=
ν x^{⊗ D}
+
λ\sum_{j=1}^{D}
 x^{⊗(j-1)}⊗ u⊗ x^{⊗(D-j)},
````

and therefore the induced Riemannian metric is

````math
g_{(λ,x)}\bigl((ν,u),(ξ,v)\bigr)
=
νξ+Dλ^2⟨u,v⟩.
````

Thus the spherical directions are scaled by ``\sqrt{D}|λ|`` relative to
the radial direction.

The Veronese manifold is a special case of the [`Segre`](@ref) manifold: the
Veronese case uses one spherical factor repeated ``D`` times, whereas the Segre
case uses independent spherical factors. Its geometry is described in
[JacobssonSwijsenVandervekenVannieuwenhoven:2026](@cite).

# Constructor

    Veronese(n::Int, d::Int; parameter::Symbol=:type)

Generate the Veronese manifold of nonzero symmetric rank-one tensors of order
`d` over ``ℝ^n``. Both `n` and `d` must be positive. The keyword `parameter`
specifies whether they are stored in the type (`:type`) or in a field (`:field`).
"""
struct Veronese{T} <: AbstractManifold{ℝ}
    size::T
end

function Veronese(n::Int, d::Int; parameter::Symbol = :type)
    n > 0 || throw(ArgumentError("The vector-space dimension n must be positive."))
    d > 0 || throw(ArgumentError("The tensor order d must be positive."))
    size = wrap_type_parameter(parameter, (n, d))
    return Veronese{typeof(size)}(size)
end

function allocate_coordinates(::Veronese, p::Tuple, T, n::Int)
    return zeros(T, n)
end

function ManifoldsBase.allocate_result(::Veronese, ::typeof(exp), p::Tuple, X, args...)
    return (similar(p[1]), similar(p[2]))
end

function ManifoldsBase.allocate_result(M::Veronese, ::typeof(rand))
    n, _ = get_parameter(M.size)
    return (zeros(1), zeros(n))
end

function ManifoldsBase.allocate_result(M::Veronese, ::typeof(zero_vector), p)
    n, _ = get_parameter(M.size)
    T = number_eltype(p)
    return (zeros(T, 1), zeros(T, n))
end

function ManifoldsBase.allocate_result_embedding(M::Veronese, ::typeof(embed), p)
    n, d = get_parameter(M.size)
    return zeros(number_eltype(p), n^d)
end

function ManifoldsBase.allocate_result_embedding(M::Veronese, ::typeof(embed), X, p)
    n, d = get_parameter(M.size)
    T = promote_type(number_eltype(p), number_eltype(X))
    return zeros(T, n^d)
end

function ManifoldsBase.allocate_result_embedding(
        ::Veronese, ::typeof(project), A, p,
    )
    return (similar(p[1]), similar(p[2]))
end

@doc raw"""
    check_point(M::Veronese, p; kwargs...)

Check whether `p` represents a point on [`Veronese`](@ref) `M`. In the
implementation, `p = ([λ], x)` is the selected parameter pair for the tensor
``Φ(λ,x)=λ x^{⊗ D}``. The scale ``λ`` must be finite
and nonzero and ``x`` must lie on ``\mathbb S^{N-1}``.
"""
function check_point(M::Veronese, p; kwargs...)
    e = check_size(M, p)
    isnothing(e) || return e

    λ = p[1][1]
    if !isfinite(λ) || iszero(λ)
        return DomainError(λ, "The scale of a point on $M must be finite and nonzero.")
    end
    n, _ = get_parameter(M.size)
    return check_point(Sphere(n - 1; parameter = get_parameter_type(M)), p[2]; kwargs...)
end

@doc raw"""
    check_size(M::Veronese, p)
    check_size(M::Veronese, p, X)

For `M = Veronese(N, D)`, check that a stored point `p = ([λ], x)` and,
optionally, a tangent vector `X = ([ν], u)` use two-component tuple
representations with component sizes `(1,)` and `(N,)`, corresponding
respectively to their radial and spherical parts.
"""
function check_size(M::Veronese, p)
    p_size = only.(size.(p))
    n, _ = get_parameter(M.size)
    M_size = (1, n)

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $p can not belong to $M, since its size $p_size is not equal to the manifold's representation size $M_size.",
        )
    end

    return nothing
end
function check_size(M::Veronese, p, X)
    p_size = only.(size.(p))
    X_size = only.(size.(X))
    n, _ = get_parameter(M.size)
    M_size = (1, n)

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $p can not belong to $M, since its size $p_size is not equal to the manifold's representation size $M_size.",
        )
    end

    if X_size != M_size
        return DomainError(
            X_size,
            "The vector $X can not be tangent at $p on $M, since its size $X_size is not equal to the manifold's representation size $M_size.",
        )
    end

    return nothing
end

@doc raw"""
    check_vector(M::Veronese, p, X; kwargs...)

Check whether `X = ([ν], u)` is tangent at `p = ([λ], x)` on
[`Veronese`](@ref) `M`. The tangent space is

````math
T_pM\simeq\mathbb R\times T_x\mathbb S^{N-1}.
````

Thus ``ν∈ℝ`` is arbitrary and ``u`` must satisfy
``u∈T_x\mathbb S^{N-1}=x^⊥``, equivalently ``⟨x,u⟩=0``.
"""
function check_vector(M::Veronese, p, X; kwargs...)
    e = check_size(M, p, X)
    isnothing(e) || return e
    n, _ = get_parameter(M.size)
    return check_vector(
        Sphere(n - 1; parameter = get_parameter_type(M)), p[2], X[2]; kwargs...,
    )
end

@doc raw"""
    closest_representative!(M::Veronese, q, p)

Let ``p ≐ (λ,x)`` and ``q ≐ (μ,y)``. The point `q` is described by either of
the two parameter pairs

````math
(μ,y)
\quad\text{and}\quad
\bigl((-1)^D μ,-y\bigr),
````

which correspond to the same tensor.

For even ``D``, the pair whose spherical component is closest to ``x`` is
chosen. For odd ``D``, the pair whose scale has the same sign as ``λ`` is
chosen.
"""
function closest_representative!(M::Veronese, q, p)
    _, d = get_parameter(M.size)
    should_flip =
        iseven(d) ? dot(p[2], q[2]) < 0 : signbit(p[1][1]) != signbit(q[1][1])
    if should_flip
        q[1][1] *= (-1)^d
        q[2] .*= -1
    end
    return q
end

@doc raw"""
    connected_by_geodesic(M::Veronese, p, q)

Return `true` if `p` and `q` are connected by a minimizing geodesic, and
`false` otherwise.

Let ``p ≐ (λ,x)`` and let ``q ≐ (μ,y)`` use the pair selected by
[`closest_representative!`](@ref). A minimizing geodesic exists if and only if
the two points lie in the same connected component and

````math
\sqrt D\,\operatorname{dist}_{𝕊^{N-1}}(x,y) < π.
````
"""
function connected_by_geodesic(M::Veronese, p, q)
    q_closest = copy(M, q)
    closest_representative!(M, q_closest, p)
    signbit(p[1][1]) == signbit(q_closest[1][1]) || return false
    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    sphere_distance = distance(sphere, p[2], q_closest[2])
    m = sqrt(d * one(sphere_distance)) * sphere_distance
    return m < oftype(m, π)
end

function copyto!(::Veronese, q, p)
    copyto!(q[1], p[1])
    copyto!(q[2], p[2])
    return q
end

function copyto!(::Veronese, Y, p, X)
    copyto!(Y[1], X[1])
    copyto!(Y[2], X[2])
    return Y
end

@doc raw"""
    default_vector_transport_method(M::Veronese)

Return [`ProjectionTransport`](@extref `ManifoldsBase.ProjectionTransport`) as
the default vector transport method on [`Veronese`](@ref).
"""
default_vector_transport_method(::Veronese) = ProjectionTransport()

@doc raw"""
    distance(M::Veronese, p, q)

Riemannian distance between two points `p` and `q` on [`Veronese`](@ref).

Let ``p ≐ (λ,x)`` and let ``q ≐ (μ,y)`` use the pair selected by
[`closest_representative!`](@ref). Define

````math
r = |λ|,
\qquad
s = |μ|,
\qquad
m = \min\left(\sqrt D\,\operatorname{dist}_{𝕊^{N-1}}(x,y), π\right).
````

If `p` and `q` lie in the same connected component, then

````math
\operatorname{dist}(p,q)
=
\sqrt{r^2+s^2-2rs\cos(m)}
=
\sqrt{(r-s)^2+4rs\sin^2(m/2)}.
````

Otherwise, their distance is infinite. See [`connected_by_geodesic`](@ref) for
when this distance is attained by a minimizing geodesic.
"""
function distance(M::Veronese, p, q)
    q_closest = copy(M, q)
    closest_representative!(M, q_closest, p)
    signbit(p[1][1]) == signbit(q_closest[1][1]) || return Inf

    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    sphere_distance = distance(sphere, p[2], q_closest[2])
    n == 1 && !iszero(sphere_distance) && return Inf
    m = sqrt(d * one(sphere_distance)) * sphere_distance
    m = min(m, oftype(m, π))
    r = abs(p[1][1])
    s = abs(q_closest[1][1])
    return sqrt((r - s)^2 + 4 * r * s * sin(m / 2)^2)
end

@doc raw"""
    embed(M::Veronese, p)
    embed!(M::Veronese, q, p)

Embed the point with stored representative `p = ([λ], x)` into the full tensor
space using the Veronese parametrization

````math
Φ(λ,x)=λ x^{⊗ D},
\qquad
x^{⊗ D}=\underbrace{x⊗\cdots⊗ x}_{D\text{ factors}}.
````

The embedded tensor is stored as a vector of length ``N^D`` corresponding to
the full, rather than symmetry-compressed, tensor coordinates.
"""
embed(::Veronese, p)

function embed!(M::Veronese, q, p)
    _, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    if d == 1
        q .= λ .* x
    else
        q .= λ .* kron(ntuple(_ -> x, d)...)
    end
    return q
end

@doc raw"""
    embed(M::Veronese, p, X)
    embed!(M::Veronese, Y, p, X)

Let `p = ([λ], x)` be the stored representative of the point, and let
`X = ([ν], u)` represent a tangent vector at `p`. Embed `X` by applying the
differential of the Veronese parametrization,

````math
DΦ_{(λ,x)}(ν,u)
=
ν x^{⊗ D}
+
λ\sum_{j=1}^{D}
 x^{⊗(j-1)}⊗ u⊗ x^{⊗(D-j)}.
````

The first term is the radial variation and the sum contains the ``D`` ways of
inserting the spherical variation ``u`` into one tensor mode. Since
``u⊥ x``, this differential is tangent to the embedded Veronese manifold.
"""
embed(::Veronese, p, X)

function embed!(M::Veronese, Y, p, X)
    _, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    ν = X[1][1]
    u = X[2]

    if d == 1
        Y .= ν .* x .+ λ .* u
        return Y
    end

    Y .= ν .* kron(ntuple(_ -> x, d)...)
    for j in 1:d
        Y .+= λ .* kron(ntuple(i -> i == j ? u : x, d)...)
    end
    return Y
end

@doc raw"""
    exp(M::Veronese, p, X)
    exp!(M::Veronese, q, p, X)

Exponential map on [`Veronese`](@ref).

Let ``p ≐ (λ,x)`` and ``X = ([ν],u) ∈ T_pM``. Writing

````math
r = |λ|,
\qquad
\dot r = \operatorname{sign}(λ)ν,
\qquad
m = √D\,\lVert u\rVert,
````

the induced metric takes the warped-cone form

````math
g = \mathrm dr^2 + D r^2 g_{𝕊^{N-1}}.
````

Thus ``\dot r`` is the radial velocity and ``m`` is the angular speed measured
in the Veronese metric. For ``m>0``, define

````math
ρ = \sqrt{(r+\dot r)^2 + (rm)^2},
\qquad
f = \operatorname{atan}(rm,r+\dot r).
````

Then the endpoint of the geodesic with initial data ``(p,X)`` is represented by

````math
\operatorname{exp}_p(X)
≐
\left(
    \operatorname{sign}(λ)ρ,\,
    \operatorname{Exp}^{𝕊^{N-1}}_x\!\left(\frac{f}{m}u\right)
\right),
````

where ``\operatorname{Exp}^{𝕊^{N-1}}`` is the exponential map on the unit
sphere. Hence the radial component changes from ``r`` to ``ρ``, while the
spherical component moves along the sphere in the direction ``u``.

If ``m=0``, there is no spherical motion. When ``r+\dot r>0``,

````math
\operatorname{exp}_p(X) ≐ (λ+ν,x).
````

If ``r+\dot r≤0``, the radial geodesic reaches the excluded zero tensor, so the
exponential map is not defined and a `DomainError` is thrown.

The formula follows from the warped-cone geodesics in Proposition 3.1 of
[JacobssonSwijsenVandervekenVannieuwenhoven:2026](@cite).
"""
exp(::Veronese, p, X)

function exp!(M::Veronese, q, p, X)
    return exp_fused!(M, q, p, X, one(number_eltype(p)))
end

@doc raw"""
    exp_fused!(M::Veronese, q, p, X, t::Number)

Compute ``\exp_p(tX)`` and store the result in `q`.

The scalar ``t`` is incorporated directly into the exponential-map formulas,
avoiding the explicit construction of the scaled tangent vector ``tX``.
"""
function exp_fused!(M::Veronese, q, p, X, t::Number)
    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    T = number_eltype(p)
    t_point = convert(T, t)

    λ = p[1][1]
    scale_sign = sign(λ)
    r = abs(λ)
    radial_endpoint = r + scale_sign * t_point * X[1][1]
    angular_speed = sqrt(d * one(λ)) * norm(X[2])
    m = abs(t_point) * angular_speed

    if iszero(m)
        radial_endpoint > 0 || throw(
            DomainError(
                t,
                "The exponential curve reaches the zero tensor, which is not part of $M.",
            ),
        )
        q[1][1] = scale_sign * radial_endpoint
        copyto!(q[2], p[2])
        return q
    end

    radial_angular = r * m
    f = atan(radial_angular, radial_endpoint)
    q[1][1] = scale_sign * hypot(radial_endpoint, radial_angular)
    exp_fused!(sphere, q[2], p[2], X[2], t_point * f / m)
    return q
end

@doc raw"""
    get_coordinates(M::Veronese, p, X, ::DefaultOrthonormalBasis; kwargs...)

Get coordinates of ``X ≐ (ν,u)`` in ``T_p\mathcal V_{N,D}`` using a
[`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`).

For ``p ≐ (λ,x)``, let ``c_{\mathbb S}(u)`` denote the orthonormal coordinates
of ``u ∈ T_x\mathbb S^{N-1}``. Since the spherical part of the Veronese metric
is scaled by ``Dλ^2``, the coordinates are

````math
c
=
\begin{bmatrix}
    ν \\
    √D\,|λ|\,c_{\mathbb S}(u)
\end{bmatrix}.
````
"""
get_coordinates(M::Veronese, p, X, ::DefaultOrthonormalBasis; kwargs...)

function get_coordinates_orthonormal!(
        M::Veronese, c, p, X, ::RealNumbers; kwargs...,
    )
    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    c[1] = X[1][1]
    get_coordinates_orthonormal!(sphere, view(c, 2:n), p[2], X[2], ℝ; kwargs...)
    c[2:n] .*= sqrt(d * one(p[1][1])) * abs(p[1][1])
    return c
end

@doc raw"""
    get_embedding(M::Veronese)

For `M = Veronese(N, D)`, return the Euclidean ambient space containing
[`Veronese`](@ref), represented as ``\mathbb R^{N^D}``. Although points of the
manifold are symmetric tensors, the current embedding uses all ``N^D`` tensor
coordinates rather than a symmetry-compressed basis.
"""
function get_embedding(M::Veronese)
    n, d = get_parameter(M.size)
    return Euclidean(n^d; parameter = get_parameter_type(M))
end

get_parameter_type(::Veronese{<:TypeParameter}) = :type
get_parameter_type(::Veronese{Tuple{Int, Int}}) = :field

@doc raw"""
    get_vector(M::Veronese, p, c, ::DefaultOrthonormalBasis; kwargs...)

Get the tangent vector ``X∈T_p\mathcal V_{N,D}`` from coordinates `c` using a
[`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`).
For ``p≐(λ,x)``, the first coordinate is the radial component and the remaining
coordinates are divided by ``\sqrt D\,|λ|`` before being converted to a tangent
vector in ``T_x\mathbb S^{N-1}``.
"""
get_vector(M::Veronese, p, c, ::DefaultOrthonormalBasis; kwargs...)

function get_vector_orthonormal!(
        M::Veronese, X, p, c, ::RealNumbers; kwargs...,
    )
    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    X[1][1] = c[1]
    get_vector_orthonormal!(sphere, X[2], p[2], view(c, 2:n), ℝ; kwargs...)
    X[2] ./= sqrt(d * one(p[1][1])) * abs(p[1][1])
    return X
end

@doc raw"""
    inner(M::Veronese, p, X, Y)

Inner product between tangent vectors ``X≐(ν,u)`` and ``Y≐(ξ,v)`` at
``p≐(λ,x)``. The Frobenius metric of the symmetric tensor embedding induces

````math
⟨X,Y⟩_p
=
νξ+Dλ^2⟨u,v⟩.
````

Here ``ν,ξ∈ℝ`` and ``u,v∈T_x\mathbb S^{N-1}=x^⊥``.
"""
function inner(M::Veronese, p, X, Y)
    _, d = get_parameter(M.size)
    return X[1][1] * Y[1][1] + d * p[1][1]^2 * dot(X[2], Y[2])
end

function _isapprox(
        M::Veronese, p, q; atol = sqrt(max_eps(p[1], p[2], q[1], q[2])), kwargs...,
    )
    if is_point(M, p) && is_point(M, q)
        return isapprox(distance(M, p, q), 0; atol = atol, kwargs...)
    end
    return length(p) == length(q) &&
        all(isapprox(pi, qi; atol = atol, kwargs...) for (pi, qi) in zip(p, q))
end

function _isapprox(
        ::Veronese, p, X, Y; atol = sqrt(max_eps(X[1], X[2], Y[1], Y[2])), kwargs...,
    )
    return length(X) == length(Y) &&
        all(isapprox(Xi, Yi; atol = atol, kwargs...) for (Xi, Yi) in zip(X, Y))
end

@doc raw"""
    log(M::Veronese, p, q)
    log!(M::Veronese, X, p, q)

The logarithmic map returns the tangent vector `X` at `p` such that
``exp_p(X)=q`` along a minimizing geodesic.

Let ``p ≐ (λ,x)`` and write ``q ≐ (μ,y)`` for the pair chosen by
[`closest_representative!`](@ref). Define

````math
r = |λ|,
\qquad
s = |μ|,
\qquad
a = \operatorname{dist}_{𝕊^{N-1}}(x,y),
\qquad
m = √D\,a.
````

Here ``r`` and ``s`` are the radial magnitudes of the two points, ``a`` is the
spherical distance from ``x`` to ``y``, and ``m`` is the corresponding angular
distance in the Veronese metric.

A minimizing logarithm exists when `p` and `q` lie in the same connected
component and ``m < π``. It is given by

````math
\operatorname{log}_p(q) ≐ ([ν],u),
````

with radial component

````math
ν
=
\operatorname{sign}(λ)\bigl(s\cos(m)-r\bigr)
````

and, for ``a>0``, spherical component

````math
u
=
\frac{s}{r}
\frac{\sin(m)}{√D\,\sin(a)}
\bigl(y-\cos(a)x\bigr).
````

The vector ``u ∈ T_x𝕊^{N-1}`` points from ``x`` toward ``y`` along the sphere.
If ``a=0``, there is no spherical motion and ``u=0``.

If no minimizing geodesic exists, `log` and `log!` throw a `DomainError`.

The formula is derived in Theorem 4.4 of
[JacobssonSwijsenVandervekenVannieuwenhoven:2026](@cite).
"""
log(::Veronese, p, q)

function log!(M::Veronese, X, p, q)
    q_closest = copy(M, q)
    closest_representative!(M, q_closest, p)
    connected_by_geodesic(M, p, q_closest) || throw(
        DomainError(q, "The points are not connected by a minimizing geodesic on $M."),
    )

    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    a = distance(sphere, p[2], q_closest[2])
    m = sqrt(d * one(a)) * a
    r = abs(p[1][1])
    s = abs(q_closest[1][1])

    X[1][1] = sign(p[1][1]) * (s * cos(m) - r)
    cos_a = clamp(dot(p[2], q_closest[2]), -one(r), one(r))
    πT = oftype(m, π)
    X[2] .=
        (q_closest[2] .- cos_a .* p[2]) .* (s / r) .* sinc(m / πT) ./
        sinc(a / πT)
    X[2] .-= dot(p[2], X[2]) .* p[2]
    return X
end

@doc raw"""
    manifold_dimension(M::Veronese)

For `M = Veronese(N, D)`, return the manifold dimension ``N``. There is one
radial degree of freedom and ``N-1`` spherical degrees of freedom, so

````math
\dim\mathcal V_{N,D}=1+(N-1)=N.
````
"""
manifold_dimension(M::Veronese) = get_parameter(M.size)[1]

@doc raw"""
    project(M::Veronese, p, A)
    project!(M::Veronese, Y, p, A)

Let `p = ([λ], x)` be the stored representative of a point on
[`Veronese`](@ref). Orthogonally project an ambient tensor `A` onto the tangent
space at `p`. The ambient tensor is stored as a vector of length ``N^D`` and
the result `Y = ([ν], u)` represents a tangent vector with ``u⊥ x``.

For each mode ``j=1,\ldots,D``, let ``c_j\in\mathbb R^N`` be the contraction of
`A` with ``x`` in every mode except mode ``j``. In coordinates,

````math
(c_j)_a
=
\sum_{i_1,\ldots,i_{j-1},i_{j+1},\ldots,i_D}
A_{i_1,\ldots,i_{j-1},a,i_{j+1},\ldots,i_D}
\prod_{k\ne j}x_{i_k},
````

and set ``c=\sum_{j=1}^{D}c_j``. The radial coefficient is

````math
ν=⟨A,x^{⊗ D}⟩.
````

The spherical component is obtained from the tangent part of ``c``:

````math
u
=
\frac{(I-xx^\top)c}{Dλ}
=
\frac{c-Dν x}{Dλ}.
````

The second equality uses ``⟨x,c⟩=Dν``. The implementation
performs one final projection onto ``x^⊥`` to remove numerical roundoff.
"""
project(::Veronese, p, A)

function project!(M::Veronese, Y, p, A)
    n, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    A_tensor = reshape(A, ntuple(_ -> n, d))

    T = promote_type(number_eltype(p), number_eltype(A))
    c = zeros(T, n)
    prefix = Vector{T}(undef, d + 1)
    suffix = Vector{T}(undef, d + 1)
    ν = zero(T)

    for I in CartesianIndices(A_tensor)
        prefix[1] = one(T)
        @inbounds for j in 1:d
            prefix[j + 1] = prefix[j] * x[I[j]]
        end

        suffix[d + 1] = one(T)
        @inbounds for j in d:-1:1
            suffix[j] = suffix[j + 1] * x[I[j]]
        end

        a = A_tensor[I]
        ν += a * prefix[d + 1]
        @inbounds for j in 1:d
            c[I[j]] += a * prefix[j] * suffix[j + 1]
        end
    end

    Y[1][1] = ν
    Y[2] .= (c .- d .* ν .* x) ./ (d * λ)
    Y[2] .-= dot(x, Y[2]) .* x
    return Y
end

@doc raw"""
    Random.rand(M::Veronese; vector_at=nothing, σ=1)

Generate a random point on [`Veronese`](@ref) `M`, stored as `p = ([λ], x)`. If
`vector_at` is provided, generate a random tangent vector there, stored as
`X = ([ν], u)`. For a random point, draw ``λ`` from a standard normal
distribution (replacing an exactly zero draw by one) and independently draw
``x`` uniformly on the unit sphere by normalizing a standard Gaussian vector.
The keyword `σ` does not affect random points.

At `vector_at = ([λ], x)`, independently draw a scalar ``z`` and a vector
``g∈ℝ^N`` with independent standard normal entries, and return

````math
ν=σz,\qquad u=σ(I-xx^\top)g.
````

Thus `σ` scales both stored tangent components. Their sampling is isotropic
in the unscaled product metric; the Veronese metric weights the spherical
component by ``Dλ^2``.
"""
Random.rand(M::Veronese; vector_at = nothing, σ::Real = 1)

function Random.rand!(
        rng::AbstractRNG, M::Veronese, pX; vector_at = nothing, σ::Real = 1,
    )
    n, _ = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    if isnothing(vector_at)
        randn!(rng, pX[1])
        pX[1][1] += iszero(pX[1][1])
        rand!(rng, sphere, pX[2])
    else
        randn!(rng, pX[1])
        pX[1] .*= σ
        rand!(rng, sphere, pX[2]; vector_at = vector_at[2], σ)
    end
    return pX
end

function Base.show(io::IO, M::Veronese)
    n, d = get_parameter(M.size)
    parameter = get_parameter_type(M) === :field ? "; parameter=:field" : ""
    return print(io, "Veronese($n, $d$parameter)")
end

@doc raw"""
    vector_transport_to(M::Veronese, p, X, q, ::ProjectionTransport)

Transport a tangent vector by orthogonally projecting its ambient embedding
onto the tangent space at the destination point.

For ``M=\mathcal V_{N,D}``, let `p = ([λ], x)` and `q = ([μ], y)` be stored
point representatives, and let `X = ([ν], u)` represent a tangent vector at
`p`. Its ambient embedding is

````math
A
=
DΦ_{(λ,x)}(ν,u)
=
ν x^{⊗ D}
+
λ\sum_{j=1}^{D}
 x^{⊗(j-1)}⊗ u⊗ x^{⊗(D-j)}.
````

Projection transport returns `Y = ([ξ], v)` at `q` such that
``DΦ_{(μ,y)}(ξ,v)`` is the orthogonal projection of `A` onto the
embedded tangent space at ``Φ(μ,y)``. Set

````math
a=⟨x,y⟩,
\qquad
b=⟨u,y⟩.
````

Then

````math
ξ
=
ν a^D+Dλ b a^{D-1},
````

and, with the final term omitted when ``D=1``,

````math
c
=
Dν a^{D-1}x
+Dλ a^{D-1}u
+D(D-1)λ b a^{D-2}x,
\qquad
v=\frac{c-Dξ y}{Dμ}.
````

The implementation evaluates these contractions directly and therefore does
not form the ambient tangent vector of length ``N^D``. It handles ``D=1``
separately so that the vanishing final term does not evaluate ``a^{-1}``.
Projection transport for an embedded Riemannian submanifold is described in
Section 8.1.3 of [AbsilMahonySepulchre:2008](@cite). The formulas above
specialize this construction to the Veronese manifold by orthogonally
projecting the ambient tangent vector given by the Veronese differential onto
the tangent space at the target point, using the Euclidean-induced metric
described in [JacobssonSwijsenVandervekenVannieuwenhoven:2026](@cite).
"""
vector_transport_to(::Veronese, p, X, q, ::ProjectionTransport)

function vector_transport_to_project!(M::Veronese, Y, p, X, q)
    _, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    μ = q[1][1]
    y = q[2]
    ν = X[1][1]
    u = X[2]

    xy = dot(x, y)
    uy = dot(u, y)
    xy_dm1 = xy^(d - 1)

    ξ = ν * xy^d + d * λ * uy * xy_dm1
    Y[1][1] = ξ

    x_coefficient = d * ν * xy_dm1
    if d > 1
        x_coefficient += d * (d - 1) * λ * uy * xy^(d - 2)
    end
    u_coefficient = d * λ * xy_dm1
    Y[2] .=
        (x_coefficient .* x .+ u_coefficient .* u .- d .* ξ .* y) ./ (d * μ)
    Y[2] .-= dot(y, Y[2]) .* y
    return Y
end

@doc raw"""
    zero_vector!(M::Veronese, X, p)

Set `X` to the zero tangent vector at `p` on [`Veronese`](@ref).
"""
function zero_vector!(::Veronese, X, p)
    fill!(X[1], zero(eltype(X[1])))
    fill!(X[2], zero(eltype(X[2])))
    return X
end
