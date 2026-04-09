@doc raw"""
    Segre{𝔽,V} <: AbstractManifold{𝔽}

The Segre manifold

````math
\mathcal{S} = \operatorname{Seg}(𝔽^{n_1} \times \dots \times 𝔽^{n_d})
````

is the set of rank-one tensors in ``𝔽^{n_1} \otimes \dots \otimes 𝔽^{n_d}``.

When ``𝔽 = ℝ``, the Segre manifold is a normal Riemannian covering of

````math
\mathcal{P} = ℝ^{+} \times \mathbb{S}^{n_1 - 1} \times \dots \times \mathbb{S}^{n_d - 1}
````

equipped with a [warped product metric](https://en.wikipedia.org/wiki/Warped_product). The tuple ``(n_1, \dots, n_d)`` is called the _valence_ of the manifold.

The geometry of the Segre manifold is summarized in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite). It is named after [Corrado Segre](https://en.wikipedia.org/wiki/Corrado_Segre)(1863–1924).

# Constructor
    Segre(n::Int...; field::AbstractNumbers=ℝ)

Generate a valence `(n, ...)` Segre manifold.
`Segre(n)` is the same as ``\mathbb{R}^{n} \setminus \{ 0 \}``.
"""
struct Segre{𝔽, V} <: AbstractManifold{𝔽} end

function Segre(n::Int...; field::AbstractNumbers = ℝ)
    return Segre{field, (n...,)}()
end

function check_size(M::Segre{𝔽, V}, p) where {𝔽, V}
    p_size = only.(size.(p))
    M_size = [1, V...]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end

function check_size(M::Segre{𝔽, V}, p, X) where {𝔽, V}
    p_size = only.(size.(p))
    X_size = only.(size.(X))
    M_size = [1, V...]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    if X_size != M_size
        return DomainError(
            X_size,
            "The vector $(X) can not belong to the manifold $(M), since its size $(X_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end

@doc raw"""
    closest_representative!(M::Segre{ℝ, V}, p, q)

``\mathcal{S}`` is a ``2^d``-sheeted Riemannian covering of

````math
\mathcal{P} = ℝ^{+} \times \mathbb{S}^{n_1 - 1} \times \dots \times \mathbb{S}^{n_d - 1}
````

with a warped product metric.
Every equivalence class ``q \in \mathcal{S}`` has ``2^d`` representatives in ``\mathcal{P}``.
`closest_representative!(M, q, p)` changes representative of `q` to the one that is closest to `p` in ``\mathcal{P}``.
"""
function closest_representative!(M::Segre{ℝ, V}, q, p) where {V}

    # Find closest representation by flipping an even number of signs.
    ds = [distance(Sphere(n - 1), x, y) for (n, x, y) in zip(V, p[2:end], q[2:end])]
    flips = [false, (ds .> (pi / 2))...]
    nbr_flips = sum(flips)

    # This code is pretty ugly.
    if isodd(nbr_flips)
        if nbr_flips == length(V)
            flips[argmin(ds) + 1] = false
        else
            is = sortperm(ds; rev = true)

            flips1 = deepcopy(flips)
            flips1[is[nbr_flips] + 1] = false
            q1 = deepcopy(q)
            q1[flips1] = -q1[flips1]

            flips2 = deepcopy(flips)
            flips2[is[nbr_flips + 1] + 1] = true
            q2 = deepcopy(q)
            q2[flips2] = -q2[flips2]

            spherical_angle_sum(M, p, q1) < spherical_angle_sum(M, p, q2) ? flips = flips1 :
                flips = flips2
        end
    end

    return q[flips] = -q[flips]
end

@doc raw"""
    connected_by_geodesic(M::Segre{ℝ, V}, p, q)

``\mathcal{S}`` is not a complete manifold, i.e. not every pair `p` and `q` of points are connected by a geodesic in ``\mathcal{S}``.
`connected_by_geodesic(M, p, q)` returns `true` if two points, `p` and `q`, are connected by a geodesic, and otherwise returns `false`.
"""
function connected_by_geodesic(M::Segre{ℝ, V}, p, q) where {V}
    closest_representative!(M, q, p)
    return spherical_angle_sum(M, p, q) < pi
end

@doc raw"""
    embed(M::Segre{𝔽, V}, p)
    embed!(M::Segre{𝔽, V}, q, p)

Embed ``p ≐ (λ, x_1, …, x_d)`` in ``𝔽^{n_1 ×⋯× n_d}`` using the Kronecker product

````math
(λ, x_1, …, x_d) ↦ λ x_1 ⊗⋯⊗ x_d.
````
"""
embed(::Segre, p)

function embed!(M::Segre, q, p)
    q .= kron(p...)
    return q
end

@doc raw"""
    embed!(M::Segre{𝔽, V}, p, X)

Embed tangent vector ``X = (ν, u_1, …, u_d)`` at ``p ≐ (λ, x_1, …, x_d)`` in ``𝔽^{n_1 ×⋯× n_d}`` using the Kronecker product

````math
    (ν, u_1, …, u_d) ↦ ν x_1 ⊗⋯⊗ x_d + λ u_1 ⊗ x_2 ⊗⋯⊗ x_d + … + λ x_1 ⊗⋯⊗ x_{d - 1} ⊗ u_d.
````
"""
function embed!(::Segre{𝔽, V}, u, p, X) where {𝔽, V}
    # Product rule
    return u = sum(
        [
            kron([i == j ? xdot : x for (j, (x, xdot)) in enumerate(zip(p, X))]...) for
                (i, _) in enumerate(p)
        ]
    )
end

"""
    is_point(M::Segre{ℝ, V}, p; kwargs...)

Check whether `p` is a valid point on `M`, i.e. `p[1]` is a singleton containing a positive number and `p[i + 1]` is a point on `Sphere(V[i])`.
The tolerance can be set using the `kwargs...`.
"""
is_point(M::Segre{ℝ, V}, p; kwargs...) where {V}

function check_point(M::Segre{ℝ, V}, p; atol = 1.4901161193847656e-8, kwargs...) where {V}
    if p[1][1] <= 0.0
        return DomainError(p[1][1], "$(p) has non-positive modulus.")
    end

    for (x, n) in zip(p[2:end], V)
        e = check_point(Sphere(n - 1)::AbstractSphere, x; atol = atol, kwargs...)
        if !isnothing(e)
            return e
        end
    end
    return
end

"""
    is_vector(M::Segre{ℝ, V}, p, X, kwargs...)

Check whether `X` is a tangent vector to `p` on `M`, i.e. `X` has to be of same dimension as `p` and orthogonal to `p`.
The tolerance can be set using the `kwargs...`.
"""
is_vector(M::Segre{ℝ, V}, p, v; kwargs...) where {V}

function check_vector(M::Segre{ℝ, V}, p, X; atol = 1.4901161193847656e-8, kwargs...) where {V}
    for (x, xdot, n) in zip(p[2:end], X[2:end], V)
        e = check_vector(Sphere(n - 1)::AbstractSphere, x, xdot; atol = atol, kwargs...)
        if !isnothing(e)
            return e
        end
    end
    return
end

@doc raw"""
    distance(M::Segre{ℝ, V}, p, q)

Riemannian distance between two points `p` and `q` on the Segre manifold.

Assume ``p ≐ (λ, x_1, …, x_d)``, ``q ≐ (μ, y_1, …, y_d) ∈ \mathcal{S}`` are connected by a geodesic. Let

````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + … + \sphericalangle(x_d, y_d)^2}
````

and assume ``(μ, y_1, …, y_d)`` is the representation of ``q`` that minimizes ``m``. Then

````math
    \operatorname{dist}_{\mathcal{S}}(p, q) = \sqrt{λ^2 - 2 λμ\cos(m) + μ^2}.
````
"""
function distance(M::Segre{ℝ, V}, p, q) where {V}
    closest_representative!(M, q, p)
    m = spherical_angle_sum(M, p, q)
    return sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(m / 2)^2)
    # Equivalent to sqrt(p[1][1]^2 + q[1][1]^2 - 2 * p[1][1] * q[1][1] * cos(m)) but more stable for small m
end

@doc raw"""
    exp(M::Segre{ℝ, V}, p, X)

Exponential map on the Segre manifold.

Let ``p ≐ (λ, x_1, …, x_d) ∈ \mathcal{S}`` and ``X = (ν, u_1, …, u_d) ∈ T_p \mathcal{S}``.
The exponential map is given by

````math
    \operatorname{exp}_p(X) ≐
    \left(
        \sqrt{(λ + ν)^2 + (λ m)^2},\\
        x_1 \cos\mathopen{\Big(} \frac{f \lVert u_1 \rVert_{x_1}}{m} \mathclose{\Big)} + \frac{u_1}{\lVert u_1 \rVert_{x_1}} \sin\mathopen{\Big(} \frac{f \lVert u_1 \rVert_{x_1}}{m} \mathclose{\Big)},\\
        …,\\
        x_d \cos\mathopen{\Big(} \frac{f \lVert u_d \rVert_{x_d}}{m} \mathclose{\Big)} + \frac{u_d}{\lVert u_d \rVert_{x_d}} \sin\mathopen{\Big(} \frac{f \lVert u_d \rVert_{x_d}}{m} \mathclose{\Big)}
    \right),
````

where

````math
    \begin{aligned}
        f &= \frac{\pi}{2} - \tan^{-1}\mathopen{\Big(} \frac{λ + ν}{λ m} \mathclose{\Big)},\\
        m &= \sqrt{\lVert u_1 \rVert_{x_1}^2 + … + \lVert u_d \rVert_{x_d}^2}.
    \end{aligned}
````

If ``m = 0`` and ``-λ < ν``, then ``\operatorname{exp}_p(v) = p + X``.

The formula is derived in proposition 3.1 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
exp(M::Segre{ℝ, V}, p, X) where {V}

function exp!(::Segre{ℝ, V}, q, p, X) where {V}
    m = sqrt(
        sum(
            [
                norm(Sphere(n - 1), x, xdot)^2 for (n, x, xdot) in zip(V, p[2:end], X[2:end])
            ]
        ),
    )

    q[1][1] = sqrt((p[1][1] + X[1][1])^2 + (p[1][1] * m)^2)

    f = pi / 2 - atan((p[1][1] + X[1][1]) / (p[1][1] * m))
    if m == 0
        for (x, y) in zip(p[2:end], q[2:end])
            y .= x
        end
    else
        for (n, x, y, xdot) in zip(V, p[2:end], q[2:end], X[2:end])
            a = norm(Sphere(n - 1), x, xdot)
            y .= x * cos(a * f / m) .+ xdot * (f / m) * sinc(a * f / (m * pi))
        end
    end

    return q
end

@doc raw"""
    get_coordinates(M::Segre{𝔽, V}, p, X, ::DefaultOrthonormalBasis; kwargs...)

Get coordinates of `X` in the tangent space
``T_{(λ, x_1, …, x_d)} \mathcal{S} = \mathbb{R} × T_{x_1} \mathbb{S}^{n_1 - 1} ×…× T_{x_d} \mathbb{S}^{n_d - 1}``
using a [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) on each factor.
"""
get_coordinates(M::Segre, p, X, ::DefaultOrthonormalBasis; kwargs...)

function get_coordinates_orthonormal!(
        M::Segre{ℝ, V},
        c,
        p,
        X,
        ::RealNumbers;
        kwargs...,
    ) where {V}
    return c = vcat(
        X[1],
        p[1][1] * [
            get_coordinates(Sphere(n - 1), x, xdot, DefaultOrthonormalBasis(); kwargs...) for (n, x, xdot) in zip(V, p[2:end], X[2:end])
        ]...,
    )
end

@doc raw"""
    get_embedding(M::Segre{𝔽,V})

Return the embedding of the [`Segre`](@ref) manifold ``\mathcal{S}``, which is ``𝔽^{n_1 ×⋯× n_d}``.
"""
function get_embedding(::Segre{𝔽, V}) where {𝔽, V}
    return Euclidean(prod(V))
end

@doc raw"""
    get_vector( M::Segre{𝔽, V}, p, c, DefaultOrthonormalBasis; kwargs...)

Get tangent vector `X` from coordinates in the tangent space
``T_{(λ, x_1, …, x_d)} \mathcal{S} = \mathbb{R} × T_{x_1} \mathbb{S}^{n_1 - 1} ×⋯× T_{x_d} \mathbb{S}^{n_d - 1}``
using a [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) on each factor.
"""
get_vector(M::Segre, p, c, ::DefaultOrthonormalBasis; kwargs...)

function get_vector_orthonormal!(M::Segre{ℝ, V}, X, p, c, ::RealNumbers; kwargs...) where {V}
    c_ = deepcopy(c)
    X[1] = [c_[1]]
    c_ = c_[2:end]
    for (i, n) in enumerate(V)
        X[i + 1] =
            get_vector(
            Sphere(n - 1),
            p[i + 1],
            c_[1:(n - 1)],
            DefaultOrthonormalBasis();
            kwargs...,
        ) / p[1][1]
        c_ = c_[n:end]
    end
    return X
end

@doc raw"""
    inner(M::Segre{ℝ, V}, p, X, Y,)

Inner product between two tangent vectors ``X = (ν, u_1, …, u_d)`` and ``Y = (ξ, v_1, …, v_d)`` at ``p ≐ (λ, x_1, \dots, x_d)``.
This inner product is obtained by embedding the Segre manifold in the space of tensors equipped with the Euclidean metric:

````math
    \langle X, Y \rangle_{p} = \nu \xi + \lambda^2 (\langle u_1, v_1 \rangle_{x_1} + \dots + \langle u_d, v_d \rangle_{x_d}),
````

where ``ν, ξ ∈ T_{λ} ℝ^{+} = ℝ`` and ``u_i``, ``v_i ∈ T_{x_i} \mathbb{S}^{n_i - 1} ⊂ ℝ^{n_i}``.
"""
function inner(::Segre{ℝ}, p, X, Y)
    return X[1][1] * Y[1][1] + p[1][1]^2 * dot(X[2:end], Y[2:end])
end

@doc raw"""
    log(M::Segre{ℝ, V}, p, q)

Logarithmic map on the Segre manifold.

Let ``p ≐ (λ, x_1, …, x_d)``, ``q ≐ (μ, y_1, …, y_d) ∈ \mathcal{S}``.
Assume ``p`` and ``q`` are connected by a geodesic.

Let

````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + … + \sphericalangle(x_d, y_d)^2}
````

and assume ``(μ, y_1, …, y_d)`` is the representative of ``q`` that minimizes ``m``. Then

````math
    \operatorname{log}_p(q) =
    \left(
        \mu \cos{m} - \lambda,
        (y_1 - ⟨x_1, y_1⟩ x_1) \frac{\mu \sphericalangle(x_1, y_1) \sin{m}}{\lambda m \sin{\sphericalangle(x_1, y_1)}},
        \dots,
        (y_d - ⟨x_d, y_d⟩ x_d) \frac{\mu \sphericalangle(x_d, y_d) \sin{m}}{\lambda m \sin{\sphericalangle(x_d, y_d)}}
    \right).
````

The formula is derived in theorem 4.4 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
log(M::Segre{ℝ, V}, p, q) where {V}

function log!(M::Segre{ℝ, V}, X, p, q) where {V}
    closest_representative!(M, q, p)
    m = spherical_angle_sum(M, p, q)

    X[1][1] = q[1][1] * cos(m) - p[1][1]
    for (n, xdot, x, y) in zip(V, X[2:end], p[2:end], q[2:end])
        a = distance(Sphere(n - 1), x, y)
        xdot .= (y - dot(x, y) * x) * (q[1][1] / p[1][1]) * sinc(m / pi) / sinc(a / pi)
    end

    return X
end

manifold_dimension(::Segre{𝔽, V}) where {𝔽, V} = (1 + sum(V .- 1))

@doc raw"""
    spherical_angle_sum(M::Segre{ℝ, V}, p, q)

Let ``p ≐ (λ, x_1, …, x_d)``, ``q ≐ (μ, y_1, …, y_d) ∈ \mathcal{S}``.
Then this is

````math
    \sqrt{\sphericalangle(x_1, y_1)^2 + … + \sphericalangle(x_d, y_d)^2},
````

where ``\sphericalangle(x_i, y_i)`` is the distance between ``x_i`` and ``y_i`` on the sphere ``\mathbb{S}^{n_i - 1}``.

"""
function spherical_angle_sum(::Segre{ℝ, V}, p, q) where {V}
    return sqrt(
        sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(V, p[2:end], q[2:end])]),
    )
end

@doc raw"""
    rand(M::Segre{ℝ, V}; vector_at=nothing)

If `vector_at` is `nothing`, return a random point on

````math
    ℝ^{+} × \mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}
````

from a log-normal distribution on ``ℝ^{+}`` and a uniform distribution on ``\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}``.

If `vector_at` is not `nothing`, return a random tangent vector from a normal distribution on the tangent space.
"""
Base.rand(M::Segre{ℝ, V}; vector_at = nothing, kwargs...) where {V}

function rand!(M::Segre{ℝ, V}, pX; vector_at = nothing, kwargs...) where {V}
    if isnothing(vector_at)
        rand!(PositiveArrays(1), pX[1]; kwargs...)
        for (n, x) in zip(V, pX[2:end])
            rand!(Sphere(n - 1), x; kwargs...)
        end
        return pX
    else
        p = vector_at
        rand!(PositiveArrays(1), pX[1]; vector_at = p[1], kwargs...)
        for (n, xdot, x) in zip(V, pX[2:end], p[2:end])
            rand!(Sphere(n - 1), xdot; vector_at = x, kwargs...)
        end
        return pX
    end
end

function ManifoldsBase.allocate_result(::Segre{ℝ, V}, ::typeof(rand)) where {V}
    return [zeros(1), [zeros(n) for n in V]...]
end

@doc raw"""
    riemann_tensor(M::Segre{ℝ, V}, p, X, Y, Z)

Riemann tensor of the Segre manifold at ``p``.

``\mathcal{S}`` is locally a warped product of ``ℝ^{+}`` and ``\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}``.
If ``p ≐ (λ, x_1, …, x_d) ∈ \mathcal{S}`` and ``X``, ``Y``, ``Z ∈ T_p (\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}) ⊂ T_p \mathcal{S}``, then

````math
    R_{\mathcal{S}}(X, Y) Z = R_{\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}}(X, Y) Z + λ^{-2}(⟨X, Z⟩_p Y - ⟨Y, Z⟩_p X).
````

``R_{\mathcal{S}}`` is zero in the remaining (orthogonal) directions.
"""
function riemann_tensor(M::Segre{ℝ, V}, p, X, Y, Z) where {V}
    return [
        [0.0],
        [
            riemann_tensor(Sphere(n - 1), x, xdot1, xdot2, xdot3) for
                (n, x, xdot1, xdot2, xdot3) in zip(V, p[2:end], X[2:end], Y[2:end], Z[2:end])
        ]...,
    ] +
        (1 / p[1][1]^2) * (
        inner(M, p, [[0.0], X[2:end]...], [[0.0], Z[2:end]...]) * [[0.0], Y[2:end]...] -
            inner(M, p, [[0.0], Y[2:end]...], [[0.0], Z[2:end]...]) * [[0.0], X[2:end]...]
    )
end

@doc raw"""
    sectional_curvature(M::Segre{ℝ, V}, p, u, v)

Sectional curvature of the Segre manifold at ``p``.

``\mathcal{S}`` is locally a warped product of ``ℝ^{+}`` and ``\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}``
If ``p ≐ (λ, x_1, …, x_d) ∈ \mathcal{S}``, ``u_i ∈ T_{x_i} \mathbb{S}^{n_i - 1}``, and ``v_j ∈ T_{x_j} \mathbb{S}^{n_j - 1}``, then

````math
    K_{\mathcal{S}}(u_i, v_j) = \frac{\delta_{i j} - 1}{\lambda^2}.
````

``K_{\mathcal{S}}`` is zero in the remaining (orthogonal) directions.
"""
function sectional_curvature(M::Segre{ℝ, V}, p, X, Y) where {V}
    return inner(M, p, riemann_tensor(M, p, X, Y, Y), X) /
        (inner(M, p, X, X) * inner(M, p, Y, Y) - inner(M, p, X, Y)^2)
end

function Base.show(io::IO, ::Segre{𝔽, V}) where {𝔽, V}
    return print(io, "Segre($(join(V, ", ")); field=$(𝔽))")
end
