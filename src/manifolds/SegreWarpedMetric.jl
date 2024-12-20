@doc raw"""
    WarpedMetric{A} <: AbstractMetric

The ``A``-warped metric on the Segre manifold ``\mathcal{S}`` is a generalization of the Euclidean metric on ``\mathcal{S}``.
We denote this manifold by ``\mathcal{S}_A``.

Similarly to ``\mathcal{S}``, when ``𝔽 = ℝ``, ``\mathcal{S}_A`` is a normal Riemannian covering of the product manifold

````math
    ℝ^{+} × S^{n_1 - 1} ×⋯× S^{n_d - 1}
````

with a [warped product metric](https://en.wikipedia.org/wiki/Warped_product), but the warping function now depends on the _warping factor_ ``A``.
``A = 1`` corresponds to the usual Segre manifold.
The Segre manifold is a cone in the sense that if ``p \in \mathcal{S}``, then ``a p \in \mathcal{S}`` for all ``r \neq 0``. The tangent subspace at ``p`` defined ``\mathrm{d} (r p) / \mathrm{d} r`` is called the _radial_ direction. ``A < 1`` puts less weight on the directions orthogonal to the radial direction compared to ``\mathcal{S}``, while ``A > 1`` puts more weight on those directions.

The geometry is summarized in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).

# Constructor
    WarpedMetric(A::Real)

Generate a warped product metric with warping factor `A`.
"""
struct WarpedMetric{A} <: AbstractMetric end

function WarpedMetric(A::Real)
    return WarpedMetric{A}()
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
    distance(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, q)

Riemannian distance between two points `p` and `q` on the warped Segre manifold.

Assume ``p ≐ (λ, x_1,…, x_d)``, ``q ≐ (μ, y_1,…, y_d) ∈ \mathcal{S}_A`` are connected by a geodesic. Let

````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 +… + \sphericalangle(x_d, y_d)^2}
````

and assume ``(μ, y_1,…, y_d)`` is the representation of ``q`` that minimizes ``m``. Then

````math
    \operatorname{dist}_{\mathcal{S}_A}(p, q) = \sqrt{λ^2 - 2 λ μ \cos(A m) + μ^2}.
````
"""
function distance(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, q) where {V,A}
    closest_representative!(M, q, p)
    m = spherical_angle_sum(M, p, q)
    return sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(A * m / 2)^2)
    # Equivalent to sqrt(p[1][1]^2 + q[1][1]^2 - 2 * p[1][1] * q[1][1] * cos(A * m)) but more stable for small m
end

@doc raw"""
    exp(M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, v)

Exponential map on the warped Segre manifold.

Let ``p ≐ (λ, x_1,…, x_d) ∈ \mathcal{S}_A`` and ``v = (ν, u_1,…, u_d) ∈ T_p \mathcal{S}_A``.
Then the exponential map is given by

````math
    \operatorname{exp}_p(v) ≐
    \left(
        \sqrt{t^2 + 2 λ ν t + λ^2},\\
        x_1 \cos\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{A m} \mathclose{\Big)} + \frac{u_1}{\lVert u_1 \rVert_{x_1}} \sin\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{A m} \mathclose{\Big)},\\
       …,\\
        x_d \cos\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{A m} \mathclose{\Big)} + \frac{u_d}{\lVert u_d \rVert_{x_d}} \sin\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{A m} \mathclose{\Big)}
    \right),
````

where

````math
    \begin{aligned}
        g &= \tan^{-1}\mathopen{\Big(} t \frac{\sqrt{P^2 + 1}}{λ} + P \mathclose{\Big)} - \tan^{-1}(P),\\
        m &= \sqrt{\lVert u_1 \rVert_{x_1}^2 +… + \lVert u_d \rVert_{x_d}^2} > 0,\\
        P &= \frac{ν}{λ A m},\\
        t &= \lVert v \rVert_{p}.
    \end{aligned}
````

If ``m = 0`` and ``ν t < λ``, then ``\operatorname{exp}_p(v) = p + v``.

For a proof, see proposition 3.1 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
exp(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, v) where {V,A}

function exp!(::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, q, p, v) where {V,A}
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
    get_coordinates(M::Segre{𝔽, V}, p, v, ::DefaultOrthonormalBasis; kwargs...)

Get coordinates of `v` in the tangent space ``T_{(λ, x_1,…, x_d)} \mathcal{S}_A = \mathbb{R} × T_{x_1} S^{n_1 - 1} ×⋯× T_{x_d} S^{n_d - 1}`` using a [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) on each factor.
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
    get_vector( M::Segre{𝔽, V}, p, X, ::DefaultOrthonormalBasis; kwargs...)

Get tangent vector `v` from coordinates in the tangent space ``T_{(λ, x_1,…, x_d)} \mathcal{S}_A = \mathbb{R} × T_{x_1} S^{n_1 - 1} ×⋯× T_{x_d} S^{n_d - 1}`` using a [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) on each factor.
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
    inner(M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, u, v)

Inner product between two tangent vectors ``u = (ν, u_1,…, u_d)`` and ``v = (ξ, v_1,…, v_d)`` at ``p = (λ, x_1,…, x_d``:
````math
    ⟨u, v⟩_{p} = ν ξ + (A λ)^2 (⟨ u_1, v_1 ⟩_{x_1} +… + ⟨u_d, v_d⟩_{x_d}),
````
where ``ν``, ``ξ ∈ T_{λ} ℝ^{+} = ℝ`` and ``u_i``, ``v_i ∈ T_{x_i} S^{n_i - 1} \subset ℝ^{n_i}``.
"""
function inner(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, u, v) where {V,A}
    return u[1][1] * v[1][1] + (A * p[1][1])^2 * dot(u[2:end], v[2:end])
end

@doc raw"""
    log(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, q)

Logarithmic map on the warped Segre manifold.

Let ``p ≐ (λ, x_1,…, x_d)``, ``q ≐ (μ, y_1,…, y_d) ∈ \mathcal{S}_A``.
Assume ``p`` and ``q`` are connected by a geodesic.
Let

````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 +… + \sphericalangle(x_d, y_d)^2}
````

and assume ``(μ, y_1,…, y_d)`` is the representation of ``q`` that minimizes ``m``. Then

````math
    \operatorname{log}_p(q) =
    c \left(
        \frac{λ A m \mathopen{\Big(} \cos(A m) - \frac{λ}{μ} \mathclose{\Big)}}{\sin(A m)},
        \frac{\sphericalangle(x_1, y_1) (y_1 - ⟨ x_1, y_1 ⟩ x_1)}{\sin(\sphericalangle(x_1, y_1))},
       …,
        \frac{\sphericalangle(x_d, y_d) (y_d - ⟨ x_d, y_d ⟩ x_d)}{\sin(\sphericalangle(x_d, y_d))}
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
    riemann_tensor(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, u, v)

Riemann tensor of the warped Segre manifold at ``p``.

``\mathcal{S}_A`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} ×⋯× S^{n_d - 1}``. If ``p ≐ (λ, x_1,…, x_d) ∈ \mathcal{S}_A`` and ``u``, ``v``, ``w ∈ T_{(x_1,…, x_d)} (S^{n_1 - 1} ×⋯× S^{n_d - 1}) \subset T_p \mathcal{S}_A`` then
````math
    R_{\mathcal{S}_A}(u, v) w = R_{S^{n_1 - 1} ×⋯× S^{n_d - 1}}(u, v) w + λ^{-2}(⟨ u, w ⟩_{p} v - ⟨ v, w ⟩_{p} u).
````
``R_{\mathcal{S}_A}`` is zero in the remaining (orthogonal) directions.
"""
function riemann_tensor(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    p,
    X,
    Y,
    Z,
) where {V,A}
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
    sectional_curvature(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, u, v)

Sectional curvature of the warped Segre manifold at ``p``.

``\mathcal{S}_A`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} ×⋯× S^{n_d - 1}``
If ``p = (λ, x_1,…, x_d) ∈ \mathcal{S}``, ``u_i ∈ T_{x_i} S^{n_i - 1}``, and ``v_j ∈ T_{x_j} S^{n_j - 1}``, then
````math
    K_{\mathcal{S}_A}(u_i, v_j) = -(1 - \delta_{i j}) λ^2.
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

function show(io::IO, ::WarpedMetric{A}) where {A}
    return print(io, "WarpedMetric($A)")
end

function spherical_angle_sum(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    p,
    q,
) where {V,A}
    return spherical_angle_sum(M.manifold, p, q)
end
