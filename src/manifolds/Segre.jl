@doc raw"""
    Segre{𝔽,V} <: AbstractManifold{𝔽}

The Segre manifold

````math
    \mathcal{S} = \operatorname{Seg}(𝔽^{n_1} \times \dots \times 𝔽^{n_d})
````

is the set of rank-one tensors in ``𝔽^{n_1} \otimes \dots \otimes 𝔽^{n_d}``.

When ``𝔽 = ℝ``, the Segre manifold is a normal Riemannian covering of

````math
    \mathcal{P} = ℝ^{+} \times S^{n_1 - 1} \times \dots \times S^{n_d - 1}
````

with the [warped product metric](https://en.wikipedia.org/wiki/Warped_product) [`inner`](@ref inner(::Segre, ::Any)). The tuple ``(n_1, \dots, n_d)`` is called the _valence_ of the manifold.

The geometry is summarized in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).

The manifold is named after [Beniamino Segre](https://en.wikipedia.org/wiki/Beniamino_Segre)(1903–1977).

# Constructor
    Segre(n::Int...; field::AbstractNumbers=ℝ)

Generate a valence `(n, ...)` Segre manifold.
`Segre(n)` is the same as ``\mathbb{R} \setminus \{ 0 \}``.
"""
struct Segre{𝔽,V} <: AbstractManifold{𝔽} end

function Segre(n::Int...; field::AbstractNumbers=ℝ)
    return Segre{field,(n...,)}()
end

manifold_dimension(::Segre{𝔽,V}) where {𝔽,V} = (1 + sum(V .- 1))

"""
    check_size(M::Segre{𝔽, V}, p)

Check whether `p` has the right size for `Segre` manifold `M`.
"""
function check_size(M::Segre{𝔽,V}, p;) where {𝔽,V}
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

"""
    check_size(M::Segre{𝔽, V}, p, v)

Check whether `p` and `v` have the right size for `Segre` manifold `M`.
"""
function check_size(M::Segre{𝔽,V}, p, v;) where {𝔽,V}
    p_size = only.(size.(p))
    v_size = only.(size.(v))
    M_size = [1, V...]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    if v_size != M_size
        return DomainError(
            v_size,
            "The vector $(v) can not belong to the manifold $(M), since its size $(v_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end

"""
    check_point(M::Segre{ℝ, V}, p; kwargs...)

Check whether `p` is a valid point on `M`, i.e. `p[1]` is a singleton containing a positive number and `p[i + 1]` is a point on `Sphere(V[i])`. The tolerance can be set using the `kwargs...`.
"""
function check_point(M::Segre{ℝ,V}, p; atol=1.4901161193847656e-8, kwargs...) where {V}
    if p[1][1] <= 0.0
        return DomainError(p[1][1], "$(p) has non-positive modulus.")
    end

    for (x, n) in zip(p[2:end], V)
        e = check_point(Sphere(n - 1)::AbstractSphere, x; atol=atol, kwargs...)
        if !isnothing(e)
            return e
        end
    end
end

"""
    function check_vector(M::Segre{ℝ, V}, p, v, kwargs...)

Check whether `v` is a tangent vector to `p` on `M`, i.e. after `check_point(M, p)`, `v` has to be of same dimension as `p` and orthogonal to `p`. The tolerance can be set using the `kwargs...`.
"""
function check_vector(M::Segre{ℝ,V}, p, v; atol=1.4901161193847656e-8, kwargs...) where {V}
    for (x, xdot, n) in zip(p[2:end], v[2:end], V)
        e = check_vector(Sphere(n - 1)::AbstractSphere, x, xdot; atol=atol, kwargs...)
        if !isnothing(e)
            return e
        end
    end
end

@doc raw"""
    function embed(M::Segre{𝔽, V}, p)
    function embed!(M::Segre{𝔽, V}, q, p)

Embed ``p ≐ (λ, x_1, …, x_d)`` in ``𝔽^{n_1 ×⋯× n_d}`` using the Kronecker product:
````math
    (λ, x_1, …, x_d) ↦ λ x_1 ⊗⋯⊗ x_d.
````
"""
embed(::Segre, p)

function embed!(M::Segre, q, p)
    return q = kron(p...)
end

@doc raw"""
    embed!(M::Segre{𝔽, V}, p, v)

Embed tangent vector ``v = (ν, u_1, …, u_d)`` at ``p ≐ (λ, x_1, …, x_d)`` in ``𝔽^{n_1 ×⋯× n_d}`` using the Kronecker product:

````math
    (ν, u_1, …, u_d) ↦ ν x_1 ⊗⋯⊗ x_d + λ u_1 ⊗ x_2 ⊗⋯⊗ x_d + … + λ x_1 ⊗⋯⊗ x_{d - 1} ⊗ u_d.
````
"""
function embed!(::Segre{𝔽,V}, u, p, v) where {𝔽,V}
    # Product rule
    return u = sum([
        kron([i == j ? xdot : x for (j, (x, xdot)) in enumerate(zip(p, v))]...) for
        (i, _) in enumerate(p)
    ])
end

@doc raw"""
    function get_coordinates(M::Segre{𝔽, V}, p, v, ::DefaultOrthonormalBasis; kwargs...)

Get coordinates of `v` in the tangent space
``T_{(λ, x_1, …, x_d)} \mathcal{S} = \mathbb{R} × T_{x_1} S^{n_1 - 1} ×…× T_{x_d} S^{n_d - 1}``
using a `DefaultOrthonormalBasis` on each factor.
"""
get_coordinates(M::Segre, p, v, ::DefaultOrthonormalBasis; kwargs...)

function get_coordinates_orthonormal!(
    M::Segre{ℝ,V},
    X,
    p,
    v,
    ::RealNumbers;
    kwargs...,
) where {V}
    return X = vcat(
        v[1],
        p[1][1] * [
            get_coordinates(Sphere(n - 1), x, xdot, DefaultOrthonormalBasis(); kwargs...) for (n, x, xdot) in zip(V, p[2:end], v[2:end])
        ]...,
    )
end

@doc raw"""
    function get_vector( M::Segre{𝔽, V}, p, X, DefaultOrthonormalBasis; kwargs...)

Get tangent vector `v` from coordinates in the tangent space
``T_{(λ, x_1, …, x_d)} \mathcal{S} = \mathbb{R} × T_{x_1} S^{n_1 - 1} ×⋯× T_{x_d} S^{n_d - 1}``
using `DefaultOrthonormalBasis` on each factor.
"""
get_vector(M::Segre, p, X, ::DefaultOrthonormalBasis; kwargs...)

function get_vector_orthonormal!(M::Segre{ℝ,V}, v, p, X, ::RealNumbers; kwargs...) where {V}
    X_ = deepcopy(X)
    v[1] = [X_[1]]
    X_ = X_[2:end]
    for (i, n) in enumerate(V)
        v[i + 1] =
            get_vector(
                Sphere(n - 1),
                p[i + 1],
                X_[1:(n - 1)],
                DefaultOrthonormalBasis();
                kwargs...,
            ) / p[1][1]
        X_ = X_[n:end]
    end
    return v
end

@doc raw"""
    function inner(M::Segre{ℝ, V}, p, u, v,)

Inner product between two tangent vectors ``u = (ν, u_1, …, u_d)`` and ``v = (ξ, v_1, …, v_d)`` at ``p ≐ (λ, x_1, \dots, x_d)``.
This inner product is obtained by embedding the Segre manifold in the space of tensors equipped with the Euclidean metric:

````math
    \langle u, v \rangle_{p} = \nu \xi + \lambda^2 (\langle u_1, v_1 \rangle_{x_1} + \dots + \langle u_d, v_d \rangle_{x_d}),
````

where ``ν, ξ ∈ T_{λ} ℝ^{+} = ℝ`` and ``u_i``, ``v_i ∈ T_{x_i} S^{n_i - 1} ⊂ ℝ^{n_i}``.
"""
function inner(::Segre{ℝ}, p, u, v)
    return u[1][1] * v[1][1] + p[1][1]^2 * dot(u[2:end], v[2:end])
end

@doc raw"""
    function rand(M::Segre{ℝ, V}; vector_at=nothing)

If `vector_at` is `nothing`, return a random point on

````math
    ℝ^{+} × S^{n_1 - 1} ×⋯× S^{n_d - 1}
````

from a log-normal distribution on ℝ^{+} and a uniform distribution on ``S^{n_1 - 1} ×⋯× S^{n_d - 1}``.

If `vector_at` is not `nothing`, return a random tangent vector from a normal distribution on the tangent space.
"""
function rand(M::Segre{ℝ,V}; vector_at=nothing, kwargs...) where {V}
    if isnothing(vector_at)
        return [
            rand(PositiveArrays(1); kwargs...),
            [rand(Sphere(n - 1); kwargs...) for n in V]...,
        ]
    else
        return [
            rand(PositiveArrays(1); vector_at=vector_at[1], kwargs...),
            [
                rand(Sphere(n - 1); vector_at=xdot, kwargs...) for
                (xdot, n) in zip(vector_at[2:end], V)
            ]...,
        ]
    end
end

@doc raw"""
    function get_embedding(M::Segre{𝔽,V})

``\mathcal{S}`` is embedded in ``𝔽^{n_1 ×⋯× n_d}``.
"""
function get_embedding(::Segre{𝔽,V}) where {𝔽,V}
    return Euclidean(prod(V))
end

@doc raw"""
    function spherical_angle_sum(M::Segre{ℝ, V}, p, q)

Let ``p ≐ (λ, x_1, …, x_d)``, ``q ≐ (μ, y_1, …, y_d) ∈ \mathcal{S}``.
Then this is

````math
    \sqrt{\sphericalangle(x_1, y_1)^2 + … + \sphericalangle(x_d, y_d)^2},
````

where ``\sphericalangle(x_i, y_i)`` is the distance between ``x_i`` and ``y_i`` on the sphere ``S^{n_i - 1}``.

"""
function spherical_angle_sum(::Segre{ℝ,V}, p, q) where {V}
    return sqrt(
        sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(V, p[2:end], q[2:end])]),
    )
end

@doc raw"""
    function connected_by_geodesic(M::Segre{ℝ, V}, p, q)

``\mathcal{S}`` is not a complete manifold, i.e. not every pair `p` and `q` of points are connected by a geodesic in ``\mathcal{S}``.
`connected_by_geodesic(M, p, q)` returns `true` if two points, `p` and `q`, are connected by a geodesic, and otherwise returns `false`.
"""
function connected_by_geodesic(M::Segre{ℝ,V}, p, q) where {V}
    closest_representative!(M, q, p)
    return spherical_angle_sum(M, p, q) < pi
end

@doc raw"""
    function closest_representative!(M::Segre{ℝ, V}, p, q)

``\mathcal{S}`` is a ``2^d``-sheeted Riemannian covering of
````math
    \mathcal{P} = ℝ^{+} \times S^{n_1 - 1} \times \dots \times S^{n_d - 1}
````
with the metric [`inner`](@ref inner(::Segre, ::Any)).
Every equivalence class ``q \in \mathcal{S}`` has ``2^d`` representatives in ``\mathcal{P}``.
`closest_representative!(M, q, p)` changes representative of `q` to the one that is closest to `p` in ``\mathcal{P}``.
"""
function closest_representative!(M::Segre{ℝ,V}, q, p) where {V}

    # Find closest representation by flipping an even number of signs.
    ds = [distance(Sphere(n - 1), x, y) for (n, x, y) in zip(V, p[2:end], q[2:end])]
    flips = [false, (ds .> (pi / 2))...]
    nbr_flips = sum(flips)

    # This code is pretty ugly.
    if isodd(nbr_flips)
        if nbr_flips == length(V)
            flips[argmin(ds) + 1] = false
        else
            is = sortperm(ds; rev=true)

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
    function exp(M::Segre{ℝ, V}, p, v)

Exponential map on the Segre manifold.

Let ``p ≐ (λ, x_1, …, x_d) ∈ \mathcal{S}`` and ``v = (ν, u_1, …, u_d) ∈ T_p \mathcal{S}``.
Then

````math
    \operatorname{exp}_p(v) ≐
    \left(
        \sqrt{t^2 + 2 λ ν t + λ^2},\\
        x_1 \cos\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{m} \mathclose{\Big)} + \frac{u_1}{\lVert u_1 \rVert_{x_1}} \sin\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{m} \mathclose{\Big)},\\
        …,\\
        x_d \cos\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{m} \mathclose{\Big)} + \frac{u_d}{\lVert u_d \rVert_{x_d}} \sin\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{m} \mathclose{\Big)}
    \right),
````

where

````math
    g = \tan^{-1}\mathopen{\Big(} t \frac{\sqrt{P^2 + 1}}{λ} + P \mathclose{\Big)} - \tan^{-1}(P),\\
    m = \sqrt{\lVert u_1 \rVert_{x_1}^2 + … + \lVert u_d \rVert_{x_d}^2},\\
    P = \frac{\nu}{λ m},\\
    t = \lVert v \rVert_{p}.
````
If ``m = 0`` and ``ν t < λ``, then ``\operatorname{exp}_p(v) = p + v``.

For a proof, see proposition 3.1 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
exp(M::Segre{ℝ,V}, p, v) where {V}

function exp!(::Segre{ℝ,V}, q, p, v) where {V}
    m = sqrt(
        sum([
            norm(Sphere(n - 1), x, xdot)^2 for (n, x, xdot) in zip(V, p[2:end], v[2:end])
        ]),
    )

    q[1][1] = sqrt((p[1][1] + v[1][1])^2 + (p[1][1] * m)^2)

    f = pi / 2 - atan((p[1][1] + v[1][1]) / (p[1][1] * m))
    if m == 0
        for (x, y) in zip(p[2:end], q[2:end])
            y .= x
        end
    else
        for (n, x, y, xdot) in zip(V, p[2:end], q[2:end], v[2:end])
            a = norm(Sphere(n - 1), x, xdot)
            y .= x * cos(a * f / m) .+ xdot * (f / m) * sinc(a * f / (m * pi))
        end
    end

    return q
end

@doc raw"""
    function log(M::Segre{ℝ, V}, p, q)

Logarithmic map on the Segre manifold.

Let ``p ≐ (λ, x_1, …, x_d)``, ``q ≐ (μ, y_1, …, y_d) ∈ \mathcal{S}``.
Assume ``p`` and ``q`` are connected by a geodesic.

Let

````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + … + \sphericalangle(x_d, y_d)^2}
````

and assume ``(μ, y_1, …, y_d)`` is the representation of ``q`` that minimizes ``m``. Then

````math
    \operatorname{log}_p(q) =
    c \left(
        \frac{λ m \mathopen{\Big(} \operatorname{cos}(m) - \frac{λ}{μ} \mathclose{\Big)}}{\operatorname{sin}(m)},
        \frac{\sphericalangle(x_1, y_1) (y_1 - ⟨x_1, y_1⟩ x_1)}{\sin(\sphericalangle(x_1, y_1))},
        \dots,
        \frac{\sphericalangle(x_d, y_d) (y_d - ⟨x_d, y_d⟩ x_d)}{\sin(\sphericalangle(x_d, y_d))}
    \right),
````
where ``c`` is determined by ``\lVert \operatorname{log}_p(q) \rVert_{p} = \operatorname{dist}(p, q)``.

For a proof, see theorem 4.4 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
log(M::Segre{ℝ,V}, p, q) where {V}

function log!(M::Segre{ℝ,V}, v, p, q) where {V}
    closest_representative!(M, q, p)
    m = spherical_angle_sum(M, p, q)

    v[1][1] = q[1][1] * cos(m) - p[1][1]
    for (n, xdot, x, y) in zip(V, v[2:end], p[2:end], q[2:end])
        a = distance(Sphere(n - 1), x, y)
        xdot .= (y - dot(x, y) * x) * (q[1][1] / p[1][1]) * sinc(m / pi) / sinc(a / pi)
    end

    return v
end

@doc raw"""
    function distance(M::Segre{ℝ, V}, p, q)

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
function distance(M::Segre{ℝ,V}, p, q) where {V}
    closest_representative!(M, q, p)
    m = spherical_angle_sum(M, p, q)

    return sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(m / 2)^2)
    # Equivalent to sqrt(p[1][1]^2 + q[1][1]^2 - 2 * p[1][1] * q[1][1] * cos(m)) but more stable for small m
end

@doc raw"""
    function riemann_tensor(M::Segre{ℝ, V}, p, u, v, w)

Riemann tensor of the Segre manifold at ``p``.

``\mathcal{S}`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} ×⋯× S^{n_d - 1}``.
If ``p ≐ (λ, x_1, …, x_d) ∈ \mathcal{S}`` and ``u``, ``v``, ``w ∈ T_p (S^{n_1 - 1} ×⋯× S^{n_d - 1}) ⊂ T_p \mathcal{S}``, then

````math
    R_{\mathcal{S}}(u, v) w = R_{S^{n_1 - 1} ×⋯× S^{n_d - 1}}(u, v) w + λ^{-2}(⟨u,w⟩_p v - ⟨v,w⟩_p u).
````

``R_{\mathcal{S}}`` is zero in the remaining (orthogonal) directions.
"""
function riemann_tensor(M::Segre{ℝ,V}, p, u, v, w) where {V}
    return [
        [0.0],
        [
            riemann_tensor(Sphere(n - 1), x, xdot1, xdot2, xdot3) for
            (n, x, xdot1, xdot2, xdot3) in zip(V, p[2:end], u[2:end], v[2:end], w[2:end])
        ]...,
    ] +
           (1 / p[1][1]^2) * (
        inner(M, p, [[0.0], u[2:end]...], [[0.0], w[2:end]...]) * [[0.0], v[2:end]...] -
        inner(M, p, [[0.0], v[2:end]...], [[0.0], w[2:end]...]) * [[0.0], u[2:end]...]
    )
end

@doc raw"""
    function sectional_curvature(M::Segre{ℝ, V}, p, u, v)

Sectional curvature of the Segre manifold at ``p``.

``\mathcal{S}`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} ×⋯× S^{n_d - 1}``
If ``p ≐ (λ, x_1, …, x_d) ∈ \mathcal{S}``, ``u_i ∈ T_{x_i} S^{n_i - 1}``, and ``v_j ∈ T_{x_j} S^{n_j - 1}``, then

````math
    K_{\mathcal{S}}(u_i, v_j) = \frac{\delta_{i j} - 1}{\lambda^2}.
````

``K_{\mathcal{S}}`` is zero in the remaining (orthogonal) directions.
"""
function sectional_curvature(M::Segre{ℝ,V}, p, u, v) where {V}
    return inner(M, p, riemann_tensor(M, p, u, v, v), u) /
           (inner(M, p, u, u) * inner(M, p, v, v) - inner(M, p, u, v)^2)
end

function Base.show(io::IO, M::Segre{𝔽,V}) where {𝔽,V}
    return print(io, "Segre($(join(V, ", ")); field=$(𝔽))")
end
