@doc raw"""
    WarpedMetric{A} <: AbstractMetric

The ``A``-warped metric on the Segre manifold ``\mathcal{S}`` is a generalization of the Euclidean metric on ``\mathcal{S}``.
We denote this manifold by ``\mathcal{S}_A``.

Similarly to ``\mathcal{S}``, when ``𝔽 = ℝ``, ``\mathcal{S}_A`` is a normal Riemannian covering of the product manifold

````math
    ℝ^{+} × \mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}
````

with a [warped product metric](https://en.wikipedia.org/wiki/Warped_product), but the warping function now depends on the _warping factor_ ``A``.
``A = 1`` corresponds to the usual Segre manifold.
The Segre manifold is a cone in the sense that if ``p \in \mathcal{S}``, then ``r p \in \mathcal{S}`` for all ``r \neq 0``. The tangent subspace at ``p`` defined ``\mathrm{d} (r p) / \mathrm{d} r`` is called the _radial_ direction. ``A < 1`` puts less weight on the directions orthogonal to the radial direction compared to ``\mathcal{S}``, while ``A > 1`` puts more weight on those directions.

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
    exp(M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, X)

Exponential map on the warped Segre manifold.

Let ``p ≐ (λ, x_1,…, x_d) ∈ \mathcal{S}_A`` and ``X = (ν, u_1,…, u_d) ∈ T_p \mathcal{S}_A``.
Then the exponential map is given by

````math
    \operatorname{exp}_p(X) ≐
    \left(
        \sqrt{(λ + ν)^2 + (λ A m)^2},\\
        x_1 \cos\mathopen{\Big(} \frac{f \lVert u_1 \rVert_{x_1}}{A m} \mathclose{\Big)} + \frac{u_1}{\lVert u_1 \rVert_{x_1}} \sin\mathopen{\Big(} \frac{f \lVert u_1 \rVert_{x_1}}{A m} \mathclose{\Big)},\\
        …,\\
        x_d \cos\mathopen{\Big(} \frac{f \lVert u_d \rVert_{x_d}}{A m} \mathclose{\Big)} + \frac{u_d}{\lVert u_d \rVert_{x_d}} \sin\mathopen{\Big(} \frac{f \lVert u_d \rVert_{x_d}}{A m} \mathclose{\Big)}
    \right),
````

where

````math
    \begin{aligned}
        f &= \frac{\pi}{2} - \tan^{-1}\mathopen{\Big(} \frac{λ + ν}{λ A m} \mathclose{\Big)},\\
        m &= \sqrt{\lVert u_1 \rVert_{x_1}^2 + … + \lVert u_d \rVert_{x_d}^2}.
    \end{aligned}
````

If ``m = 0`` and ``-λ < ν``, then ``\operatorname{exp}_p(v) = p + X``.

The formula is derived in proposition 3.1 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
exp(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, X) where {V,A}

function exp!(::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, q, p, X) where {V,A}
    m = sqrt(
        sum([
            norm(Sphere(n - 1), x, xdot)^2 for (n, x, xdot) in zip(V, p[2:end], X[2:end])
        ]),
    )

    q[1][1] = sqrt((p[1][1] + X[1][1])^2 + (p[1][1] * A * m)^2)

    f = pi / 2 - atan((p[1][1] + X[1][1]) / (p[1][1] * A * m))
    if m == 0
        for (x, y) in zip(p[2:end], q[2:end])
            y .= x
        end
    else
        for (n, x, y, xdot) in zip(V, p[2:end], q[2:end], X[2:end])
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

Get coordinates of `X` in the tangent space ``T_{(λ, x_1,…, x_d)} \mathcal{S}_A = \mathbb{R} × T_{x_1} \mathbb{S}^{n_1 - 1} ×⋯× T_{x_d} \mathbb{S}^{n_d - 1}`` using a [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) on each factor.
"""
get_coordinates(
    M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}},
    p,
    X,
    ::DefaultOrthonormalBasis;
    kwargs...,
) where {𝔽,V,A}

function get_coordinates_orthonormal!(
    M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}},
    c,
    p,
    X,
    ::RealNumbers;
    kwargs...,
) where {𝔽,V,A}
    return c = vcat(
        X[1],
        A *
        p[1][1] *
        [
            get_coordinates(Sphere(n - 1), x, xdot, DefaultOrthonormalBasis(); kwargs...) for (n, x, xdot) in zip(V, p[2:end], X[2:end])
        ]...,
    )
end

@doc raw"""
    get_vector(M::Segre{𝔽, V}, p, c, ::DefaultOrthonormalBasis; kwargs...)

Get tangent vector `X` from coordinates in the tangent space ``T_{(λ, x_1,…, x_d)} \mathcal{S}_A = \mathbb{R} × T_{x_1} \mathbb{S}^{n_1 - 1} ×⋯× T_{x_d} \mathbb{S}^{n_d - 1}`` using a [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`) on each factor.
"""
get_vector(
    M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}},
    p,
    c,
    ::DefaultOrthonormalBasis;
    kwargs...,
) where {V,A,𝔽}

function get_vector_orthonormal!(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    X,
    p,
    c,
    ::RealNumbers;
    kwargs...,
) where {V,A}
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
            ) / (A * p[1][1])
        c_ = c_[n:end]
    end

    return X
end

@doc raw"""
    inner(M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, X, Y)

Inner product between two tangent vectors ``X = (ν, u_1,…, u_d)`` and ``Y = (ξ, v_1,…, v_d)`` at ``p \doteq (λ, x_1,…, x_d)``:
````math
    ⟨X, Y⟩_{p} = ν ξ + (A λ)^2 (⟨ u_1, v_1 ⟩_{x_1} +… + ⟨u_d, v_d⟩_{x_d}),
````
where ``ν``, ``ξ ∈ T_{λ} ℝ^{+} = ℝ`` and ``u_i``, ``v_i ∈ T_{x_i} \mathbb{S}^{n_i - 1} \subset ℝ^{n_i}``.
"""
function inner(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, X, Y) where {V,A}
    return X[1][1] * Y[1][1] + (A * p[1][1])^2 * dot(X[2:end], Y[2:end])
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

and assume ``(μ, y_1,…, y_d)`` is the representative of ``q`` that minimizes ``m``. Then

````math
    \operatorname{log}_p(q) =
    \left(
        \mu \cos{m} - \lambda,
        (y_1 - ⟨x_1, y_1⟩ x_1) \frac{\mu \sphericalangle(x_1, y_1) \sin{A m}}{\lambda A m \sin{\sphericalangle(x_1, y_1)}},
        \dots,
        (y_d - ⟨x_d, y_d⟩ x_d) \frac{\mu \sphericalangle(x_d, y_d) \sin{A m}}{\lambda A m \sin{\sphericalangle(x_d, y_d)}}
    \right).
````

The formula is derived in theorem 4.4 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
log(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, p, q) where {V,A}

function log!(M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}}, X, p, q) where {V,A}
    closest_representative!(M, q, p)
    m = spherical_angle_sum(M, p, q)

    X[1][1] = q[1][1] * cos(A * m) - p[1][1]
    for (n, xdot, x, y) in zip(V, X[2:end], p[2:end], q[2:end])
        a = distance(Sphere(n - 1), x, y)
        xdot .= (y - dot(x, y) * x) * (q[1][1] / p[1][1]) * sinc(A * m / pi) / sinc(a / pi)
    end

    return X
end

@doc raw"""
    riemann_tensor(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, X, Y)

Riemann tensor of the warped Segre manifold at ``p``.

``\mathcal{S}_A`` is locally a warped product of ``ℝ^{+}`` and ``\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}``. If ``p ≐ (λ, x_1,…, x_d) ∈ \mathcal{S}_A`` and ``X``, ``Y``, ``Z ∈ T_{(x_1,…, x_d)} (\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}) \subset T_p \mathcal{S}_A`` then
````math
    R_{\mathcal{S}_A}(X, Y) Z = R_{\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}}(X, Y) Z + λ^{-2}(⟨ X, Z ⟩_{p} Y - ⟨ Y, Z ⟩_{p} X).
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
    sectional_curvature(M::MetricManifold{ℝ, Segre{ℝ,V}, WarpedMetric{A}}, p, X, Y)

Sectional curvature of the warped Segre manifold at ``p``.

``\mathcal{S}_A`` is locally a warped product of ``ℝ^{+}`` and ``\mathbb{S}^{n_1 - 1} ×⋯× \mathbb{S}^{n_d - 1}``
If ``p = (λ, x_1,…, x_d) ∈ \mathcal{S}``, ``u_i ∈ T_{x_i} \mathbb{S}^{n_i - 1}``, and ``v_j ∈ T_{x_j} \mathbb{S}^{n_j - 1}``, then
````math
    K_{\mathcal{S}_A}(u_i, v_j) = \frac{A^{-2} \delta_{i j} - 1}{λ^{2}}.
````
``K_{\mathcal{S}_A}`` is zero in the remaining (orthogonal) directions.
"""
function sectional_curvature(
    M::MetricManifold{ℝ,Segre{ℝ,V},WarpedMetric{A}},
    p,
    X,
    Y,
) where {V,A}
    return inner(M, p, riemann_tensor(M, p, X, Y, Y), X) /
           (inner(M, p, X, X) * inner(M, p, Y, Y) - inner(M, p, X, Y)^2)
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
