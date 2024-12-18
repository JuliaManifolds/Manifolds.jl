@doc raw"""
    WarpedMetric{A} <: AbstractMetric

The ``A``-warped metric on the Segre manifold ``\mathcal{S}`` is a generalization of the Euclidean metric on ``\mathcal{S}``.
We this manifold by ``\mathcal{S}_A``.

Similarly to ``\mathcal{S}``, when ``𝔽 = ℝ``, ``\mathcal{S}_A`` is a normal Riemannian covering of the product manifold
````math
    ℝ^{+} \times S^{n_1 - 1} \times \dots \times S^{n_d - 1}
````
with a [warped product metric](https://en.wikipedia.org/wiki/Warped_product), but the warping function now depends on the _warping factor_ ``A``.
``A = 1`` corresponds to the usual Segre manifold.

The geometry is summarized in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
struct WarpedMetric{A} <: AbstractMetric end

@doc raw"""
    function get_coordinates(M::Segre{𝔽, V}, p, v, ::DefaultOrthonormalBasis; kwargs...)

Get coordinates of `v` in the tangent space ``T_{(\lambda, x_1, \dots, x_d)} \mathcal{S}_A = \mathbb{R} \times T_{x_1} S^{n_1 - 1} \times \dots \times T_{x_d} S^{n_d - 1}`` using `DefaultOrthonormalBasis` on each factor.
"""
get_coordinates(
    M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}},
    p,
    v,
    ::DefaultOrthonormalBasis;
    kwargs...,
) where {𝔽,V,A}

function get_coordinates_orthonormal!(
    M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}},
    X,
    p,
    v,
    ::RealNumbers;
    kwargs...,
) where {𝔽,V,A}
    return X = vcat(
        v[1],
        A *
        p[1][1] *
        [
            get_coordinates(Sphere(n - 1), x, xdot, DefaultOrthonormalBasis(); kwargs...) for (n, x, xdot) in zip(V, p[2:end], v[2:end])
        ]...,
    )
end

@doc raw"""
    function get_vector( M::Segre{𝔽, V}, p, X, ::DefaultOrthonormalBasis; kwargs...)

Get tangent vector `v` from coordinates in the tangent space ``T_{(\lambda, x_1, \dots, x_d)} \mathcal{S}_A = \mathbb{R} \times T_{x_1} S^{n_1 - 1} \times \dots \times T_{x_d} S^{n_d - 1}`` using `DefaultOrthonormalBasis` on each factor.
"""
get_vector(
    M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}},
    p,
    X,
    ::DefaultOrthonormalBasis;
    kwargs...,
) where {V,A,𝔽}

function get_vector_orthonormal!(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    v,
    p,
    X,
    ::RealNumbers;
    kwargs...,
) where {V,A}
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
            ) / (A * p[1][1])
        X_ = X_[n:end]
    end

    return v
end

@doc raw"""
    function inner( M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, u, v)

Inner product between two tangent vectors ``u = (\nu, u_1, \dots, u_d)`` and ``v = (\xi, v_1, \dots, v_d)`` at ``p = (\lambda, x_1, \dots, x_d``:
````math
    \langle u, v \rangle_{p} = \nu \xi + (A \lambda)^2 (\langle u_1, v_1 \rangle_{x_1} + \dots + \langle u_d, v_d \rangle_{x_d}),
````
where ``\nu``, ``\xi \in T_{\lambda} ℝ^{+} = ℝ`` and ``u_i``, ``v_i \in T_{x_i} S^{n_i - 1} \subset ℝ^{n_i}``.
"""
function inner(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, u, v) where {V,A}
    return u[1][1] * v[1][1] + (A * p[1][1])^2 * dot(u[2:end], v[2:end])
end

function spherical_angle_sum(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    p,
    q,
) where {V,A}
    return spherical_angle_sum(M.manifold, p, q)
end

function connected_by_geodesic(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    p,
    q,
) where {V,A}
    return connected_by_geodesic(M.manifold, p, q)
end

function closest_representative!(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    q,
    p,
) where {V,A}
    return closest_representative!(M.manifold, q, p)
end

@doc raw"""
    function exp(M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, v)

Exponential map on the warped Segre manifold.

Let ``p \doteq (\lambda, x_1, \dots, x_d) \in \mathcal{S}_A`` and ``v = (\nu, u_1, \dots, u_d) \in T_p \mathcal{S}_A``.
Then
````math
    \operatorname{exp}_p(v) \doteq
    \left(
        \sqrt{t^2 + 2 \lambda \nu t + \lambda^2},\\
        x_1 \cos\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{A m} \mathclose{\Big)} + \frac{u_1}{\lVert u_1 \rVert_{x_1}} \sin\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{A m} \mathclose{\Big)},\\
        \dots,\\
        x_d \cos\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{A m} \mathclose{\Big)} + \frac{u_d}{\lVert u_d \rVert_{x_d}} \sin\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{A m} \mathclose{\Big)}
    \right),
````
where
````math
    g = \tan^{-1}\mathopen{\Big(} t \frac{\sqrt{P^2 + 1}}{\lambda} + P \mathclose{\Big)} - \tan^{-1}(P),\\
    m = \sqrt{\lVert u_1 \rVert_{x_1}^2 + \dots + \lVert u_d \rVert_{x_d}^2},\\
    P = \frac{\nu}{\lambda A m},\\
    t = \lVert v \rVert_{p}.
````
If ``m = 0`` and ``\nu t < \lambda``, then ``\operatorname{exp}_p(v) = p + v``.

For a proof, see proposition 3.1 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
exp(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, v) where {V,A}

function exp!(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, q, p, v) where {V,A}
    m = sqrt(
        sum([
            norm(Sphere(n - 1), x, xdot)^2 for (n, x, xdot) in zip(V, p[2:end], v[2:end])
        ]),
    )

    q[1][1] = sqrt((p[1][1] + v[1][1])^2 + (p[1][1] * A * m)^2)

    f = pi / 2 - atan((p[1][1] + v[1][1]) / (p[1][1] * A * m))
    if m == 0
        for (x, y) in zip(p[2:end], q[2:end])
            y .= x
        end
    else
        for (n, x, y, xdot) in zip(V, p[2:end], q[2:end], v[2:end])
            a = norm(Sphere(n - 1), x, xdot)
            y .=
                x * cos(a * f / (A * m)) .+
                xdot * (f / (A * m)) * sinc(a * f / (A * m * pi))
        end
    end

    return q
end

@doc raw"""
    function log(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, q)

Logarithmic map on the warped Segre manifold.

Let ``p \doteq (\lambda, x_1, \dots, x_d)``, ``q \doteq (\mu, y_1, \dots, y_d) \in \mathcal{S}_A``.
Assume ``p`` and ``q`` are connected by a geodesic.
Let
````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + \dots + \sphericalangle(x_d, y_d)^2}
````
and assume ``(\mu, y_1, \dots, y_d)`` is the representation of ``q`` that minimizes ``m``. Then
````math
    \operatorname{log}_p(q) =
    c \left(
        \frac{\lambda A m \mathopen{\Big(} \cos(A m) - \frac{\lambda}{\mu} \mathclose{\Big)}}{\sin(A m)},
        \frac{\sphericalangle(x_1, y_1) (y_1 - \langle x_1, y_1 \rangle x_1)}{\sin(\sphericalangle(x_1, y_1))},
        \dots,
        \frac{\sphericalangle(x_d, y_d) (y_d - \langle x_d, y_d \rangle x_d)}{\sin(\sphericalangle(x_d, y_d))}
    \right),
````
where ``c`` is determined by ``\lVert \operatorname{log}_p(q) \rVert_{p} = \operatorname{dist}(p, q)``.

For a proof, see theorem 4.4 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
log(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, q) where {V,A}

function log!(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, v, p, q) where {V,A}
    closest_representative!(M, q, p)
    m = spherical_angle_sum(M, p, q)

    v[1][1] = q[1][1] * cos(A * m) - p[1][1]
    for (n, xdot, x, y) in zip(V, v[2:end], p[2:end], q[2:end])
        a = distance(Sphere(n - 1), x, y)
        xdot .= (y - dot(x, y) * x) * (q[1][1] / p[1][1]) * sinc(A * m / pi) / sinc(a / pi)
    end

    return v
end

@doc raw"""
    function distance(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, q)

Riemannian distance between two points `p` and `q` on the warped Segre manifold.

Assume ``p \doteq (\lambda, x_1, \dots, x_d)``, ``q \doteq (\mu, y_1, \dots, y_d) \in \mathcal{S}_A`` are connected by a geodesic. Let
````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + \dots + \sphericalangle(x_d, y_d)^2}
````
and assume ``(\mu, y_1, \dots, y_d)`` is the representation of ``q`` that minimizes ``m``. Then
````math
    \operatorname{dist}_{\mathcal{S}_A}(p, q) = \sqrt{\lambda^2 - 2 \lambda \mu \cos(A m) + \mu^2}.
````
"""
function distance(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, q) where {V,A}
    closest_representative!(M, q, p)
    m = spherical_angle_sum(M, p, q)
    return sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(A * m / 2)^2)
    # Equivalent to sqrt(p[1][1]^2 + q[1][1]^2 - 2 * p[1][1] * q[1][1] * cos(A * m)) but more stable for small m
end

@doc raw"""
    function riemann_tensor(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, u, v)

Riemann tensor of the warped Segre manifold at ``p``.

``\mathcal{S}_A`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}``. If ``p \doteq (\lambda, x_1, \dots, x_d) \in \mathcal{S}_A`` and ``u``, ``v``, ``w \in T_{(x_1, \dots, x_d)} (S^{n_1 - 1} \times \dots \times S^{n_d - 1}) \subset T_p \mathcal{S}_A`` then
````math
    R_{\mathcal{S}_A}(u, v) w = R_{S^{n_1 - 1} \times \dots \times S^{n_d - 1}}(u, v) w + \lambda^{-2}(\langle u, w \rangle_{p} v - \langle v, w \rangle_{p} u).
````
``R_{\mathcal{S}_A}`` is zero in the remaining (orthogonal) directions.
"""
function riemann_tensor(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    p,
    u,
    v,
    w,
) where {V,A}
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
    function sectional_curvature(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, u, v)

Sectional curvature of the warped Segre manifold at ``p``.

``\mathcal{S}_A`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}``
If ``p = (\lambda, x_1, \dots, x_d) \in \mathcal{S}``, ``u_i \in T_{x_i} S^{n_i - 1}``, and ``v_j \in T_{x_j} S^{n_j - 1}``, then
````math
    K_{\mathcal{S}_A}(u_i, v_j) = -(1 - \delta_{i j}) \lambda^2.
````
``K_{\mathcal{S}_A}`` is zero in the remaining (orthogonal) directions.
"""
function sectional_curvature(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    p,
    u,
    v,
) where {V,A}
    return inner(M, p, riemann_tensor(M, p, u, v, v), u) /
           (inner(M, p, u, u) * inner(M, p, v, v) - inner(M, p, u, v)^2)
end
