@doc raw"""
    det_local_metric(M::AbstractManifold, p, B::AbstractBasis)

Return the determinant of local matrix representation of the metric tensor ``g``, i.e. of the
matrix ``G(p)`` representing the metric in the tangent space at ``p`` with as a matrix.

See also [`local_metric`](@ref)

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`det_local_metric`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
function det_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    @warn "`det_local_metric(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `det_local_metric(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
    return det(local_metric(M, p, B))
end

"""
    einstein_tensor(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = default_differential_backend())

Compute the Einstein tensor of the manifold `M` at the point `p`, see [https://en.wikipedia.org/wiki/Einstein_tensor](https://en.wikipedia.org/wiki/Einstein_tensor)

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`einstein_tensor`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
function einstein_tensor(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    @warn "`einstein_tensor(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `einstein_tensor(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
    Ric = ricci_tensor(M, p, B; backend = backend)
    g = local_metric(M, p, B)
    Ginv = inverse_local_metric(M, p, B)
    S = sum(Ginv .* Ric)
    G = Ric - g .* S / 2
    return G
end
@trait_function einstein_tensor(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)


@doc raw"""
    inverse_local_metric(M::AbstractManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation of the inverse metric (cometric) tensor
of the tangent space at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` with respect
to the [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) basis `B`.

The metric tensor (see [`local_metric`](@ref)) is usually denoted by ``G = (g_{ij}) ∈ 𝔽^{d×d}``,
where ``d`` is the dimension of the manifold.

Then the inverse local metric is denoted by ``G^{-1} = g^{ij}``.

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`inverse_local_metric`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
inverse_local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
function inverse_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    @warn "`inverse_local_metric(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `inverse_local_metric(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
    return inv(local_metric(M, p, B))
end
@trait_function inverse_local_metric(M::AbstractDecoratorManifold, p, B::AbstractBasis)


@doc raw"""
    local_metric(M::AbstractManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation at the point `p` of the metric tensor ``g`` with
respect to the [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `B` on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M`.
Let ``d``denote the dimension of the manifold and $b_1,\ldots,b_d$ the basis vectors.
Then the local matrix representation is a matrix ``G\in 𝔽^{n×n}`` whose entries are
given by ``g_{ij} = g_p(b_i,b_j), i,j\in\{1,…,d\}``.

This yields the property for two tangent vectors (using Einstein summation convention)
``X = X^ib_i, Y=Y^ib_i \in T_p\mathcal M`` we get ``g_p(X, Y) = g_{ij} X^i Y^j``.

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`local_metric`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
local_metric(::AbstractManifold, ::Any, ::AbstractBasis)

function local_metric(M::MetricManifold, p, B::AbstractBasis)
    @warn "`local_metric(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `local_metric(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
    (metric(M.manifold) == M.metric) && (return local_metric(M.manifold, p, B))
    return invoke(local_metric, Tuple{AbstractManifold, Any, AbstractBasis}, M, p, B)
end

@doc raw"""
    local_metric_jacobian(M::AbstractManifold, p, B::AbstractBasis;
        backend::AbstractDiffBackend,
    )

Get partial derivatives of the local metric of `M` at `p` in basis `B` with respect to the
coordinates of `p`, ``\frac{∂}{∂ p^k} g_{ij} = g_{ij,k}``. The
dimensions of the resulting multi-dimensional array are ordered ``(i,j,k)``.

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based [`christoffel_symbols_first`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)`
    or [`christoffel_symbols_second`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead,
    which handle derivatives of the local metric internally.
"""
local_metric_jacobian(::AbstractManifold, ::Any, B::AbstractBasis)
function local_metric_jacobian(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    @warn "`local_metric_jacobian(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `christoffel_symbols_second(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
    n = size(p, 1)
    ∂g = reshape(_jacobian(q -> local_metric(M, q, B), copy(M, p), backend), n, n, n)
    return ∂g
end

@doc raw"""
    log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)

Return the natural logarithm of the metric density ``ρ`` of `M` at `p`. The density
is given by ``ρ = \sqrt{|\det [g_{ij}]|}`` for the metric tensor expressed in basis `B`.

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`log_local_metric_density`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
log_local_metric_density(::AbstractManifold, ::Any, ::AbstractBasis)
function log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)
    @warn "`log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `log_local_metric_density(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
    return log(abs(det_local_metric(M, p, B))) / 2
end

function norm(M::MetricManifold, p, X::TFVector)
    return sqrt(dot(X.data, local_metric(M, p, X.basis) * X.data))
end

@doc raw"""
    ricci_curvature(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = default_differential_backend())

Compute the Ricci scalar curvature of the manifold `M` at the point `p` using basis `B`.
The curvature is computed as the trace of the Ricci curvature tensor with respect to
the metric, that is ``R=g^{ij}R_{ij}`` where ``R`` is the scalar Ricci curvature at `p`,
``g^{ij}`` is the inverse local metric (see [`inverse_local_metric`](@ref)) at `p` and
``R_{ij}`` is the Ricci curvature tensor, see [`ricci_tensor`](@ref). Both the tensor and
inverse local metric are expressed in local coordinates defined by `B`, and the formula
uses the Einstein summation convention.

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`ricci_curvature`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
ricci_curvature(::AbstractManifold, ::Any, ::AbstractBasis)
function ricci_curvature(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    @warn "`ricci_curvature(M::AbstractManifold, p, B::AbstractBasis)` is deprecated. Consider using `ricci_curvature(M::AbstractManifold, A::AbstractAtlas, i, a)` instead"  maxlog = 1
    Ginv = inverse_local_metric(M, p, B)
    Ric = ricci_tensor(M, p, B; backend = backend)
    S = sum(Ginv .* Ric)
    return S
end
ManifoldsBase.@trait_function ricci_curvature(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

@doc raw"""
    christoffel_symbols_first(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )

Compute the Christoffel symbols of the first kind in local coordinates of basis `B`.
The Christoffel symbols are (in Einstein summation convention)

````math
Γ_{ijk} = \frac{1}{2} \Bigl[g_{kj,i} + g_{ik,j} - g_{ij,k}\Bigr],
````

where ``g_{ij,k}=\frac{∂}{∂ p^k} g_{ij}`` is the coordinate
derivative of the local representation of the metric tensor. The dimensions of
the resulting multi-dimensional array are ordered ``(i,j,k)``.

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`christoffel_symbols_first`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
christoffel_symbols_first(::AbstractManifold, ::Any, B::AbstractBasis)
function christoffel_symbols_first(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    ∂g = local_metric_jacobian(M, p, B; backend = backend)
    n = size(∂g, 1)
    Γ = allocate(∂g, Size(n, n, n))
    @einsum Γ[i, j, k] = 1 / 2 * (∂g[k, j, i] + ∂g[i, k, j] - ∂g[i, j, k])
    return Γ
end
@trait_function christoffel_symbols_first(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

@doc raw"""
    christoffel_symbols_second(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )

Compute the Christoffel symbols of the second kind in local coordinates of basis `B`.
For affine connection manifold the Christoffel symbols need to be explicitly implemented
while, for a [`MetricManifold`](@extref ManifoldsBase.MetricManifold) they are computed as (in Einstein summation convention)

````math
Γ^{l}_{ij} = g^{kl} Γ_{ijk},
````

where ``Γ_{ijk}`` are the Christoffel symbols of the first kind
(see [`christoffel_symbols_first`](@ref)), and ``g^{kl}`` is the inverse of the local
representation of the metric tensor. The dimensions of the resulting multi-dimensional array
are ordered ``(l,i,j)``.

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`christoffel_symbols_second`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
function christoffel_symbols_second(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    Ginv = inverse_local_metric(M, p, B)
    Γ₁ = christoffel_symbols_first(M, p, B; backend = backend)
    Γ₂ = allocate(Γ₁)
    @einsum Γ₂[l, i, j] = Ginv[k, l] * Γ₁[i, j, k]
    return Γ₂
end

@trait_function christoffel_symbols_second(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

@doc raw"""
    christoffel_symbols_second_jacobian(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )

Get partial derivatives of the Christoffel symbols of the second kind
for manifold `M` at `p` with respect to the coordinates of `B`, i.e.

```math
\frac{∂}{∂ p^l} Γ^{k}_{ij} = Γ^{k}_{ij,l}.
```

The dimensions of the resulting multi-dimensional array are ordered ``(k,i,j,l)``.

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based [`riemann_tensor`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead,
    which handles derivatives of the Christoffel symbols internally.
"""
christoffel_symbols_second_jacobian(::AbstractManifold, ::Any, B::AbstractBasis)
function christoffel_symbols_second_jacobian(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    n = size(p, 1)
    ∂Γ = reshape(
        _jacobian(q -> christoffel_symbols_second(M, q, B; backend = backend), p, backend),
        n,
        n,
        n,
        n,
    )
    return ∂Γ
end
@trait_function christoffel_symbols_second_jacobian(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

for mf in [
        christoffel_symbols_second_jacobian,
        local_metric_jacobian,
    ]
    @eval is_metric_function(::typeof($mf)) = true
end

"""
    gaussian_curvature(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = default_differential_backend())

Compute the Gaussian curvature of the manifold `M` at the point `p` using basis `B`.
This is equal to half of the scalar Ricci curvature, see [`ricci_curvature`](@ref).

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`gaussian_curvature`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
gaussian_curvature(::AbstractManifold, ::Any, ::AbstractBasis)
function gaussian_curvature(M::AbstractManifold, p, B::AbstractBasis; kwargs...)
    return ricci_curvature(M, p, B; kwargs...) / 2
end
@trait_function gaussian_curvature(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

"""
    ricci_tensor(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = default_differential_backend())

Compute the Ricci tensor, also known as the Ricci curvature tensor,
of the manifold `M` at the point `p` using basis `B`,
see [`https://en.wikipedia.org/wiki/Ricci_curvature#Introduction_and_local_definition`](https://en.wikipedia.org/wiki/Ricci_curvature#Introduction_and_local_definition).

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`ricci_tensor`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
ricci_tensor(::AbstractManifold, ::Any, ::AbstractBasis)
function ricci_tensor(M::AbstractManifold, p, B::AbstractBasis; kwargs...)
    R = riemann_tensor(M, p, B; kwargs...)
    n = size(R, 1)
    Ric = allocate(R, Size(n, n))
    @einsum Ric[i, j] = R[l, i, l, j]
    return Ric
end
@trait_function ricci_tensor(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

@doc raw"""
    riemann_tensor(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend=default_differential_backend())

Compute the Riemann tensor ``R^l_{ijk}``, also known as the Riemann curvature
tensor, at the point `p` in local coordinates defined by `B`. The dimensions of the
resulting multi-dimensional array are ordered ``(l,i,j,k)``.

The function uses the coordinate expression involving the second Christoffel symbol,
see [`https://en.wikipedia.org/wiki/Riemann_curvature_tensor#Coordinate_expression`](https://en.wikipedia.org/wiki/Riemann_curvature_tensor#Coordinate_expression)
for details.

# See also

[`christoffel_symbols_second`](@ref), [`christoffel_symbols_second_jacobian`](@ref)

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`riemann_tensor`](@ref)`(M::AbstractManifold, A::AbstractAtlas, i, a)` instead.
"""
riemann_tensor(::AbstractManifold, ::Any, ::AbstractBasis)
function riemann_tensor(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = default_differential_backend(),
    )
    n = size(p, 1)
    Γ = christoffel_symbols_second(M, p, B; backend = backend)
    ∂Γ = christoffel_symbols_second_jacobian(M, p, B; backend = backend) ./ n
    R = allocate(∂Γ, Size(n, n, n, n))
    @einsum R[l, i, j, k] =
        ∂Γ[l, i, k, j] - ∂Γ[l, i, j, k] + Γ[s, i, k] * Γ[l, s, j] - Γ[s, i, j] * Γ[l, s, k]
    return R
end
@trait_function riemann_tensor(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)

function solve_exp_ode end

@doc raw"""
    solve_exp_ode(
        M::AbstractManifold,
        p,
        X,
        t::Number;
        B::AbstractBasis = DefaultOrthonormalBasis(),
        backend::AbstractDiffBackend = default_differential_backend(),
        solver = AutoVern9(Rodas5P()),
        kwargs...,
    )

Approximate the exponential map on the manifold by evaluating the ODE describing the geodesic at 1,
assuming the default connection of the given manifold by solving the ordinary differential
equation

```math
\frac{d^2}{dt^2} p^k + Γ^k_{ij} \frac{d}{dt} p_i \frac{d}{dt} p_j = 0,
```

where ``Γ^k_{ij}`` are the Christoffel symbols of the second kind, and
the Einstein summation convention is assumed. The argument `solver` follows
the `OrdinaryDiffEq` conventions. `kwargs...` specify keyword
arguments that will be passed to `OrdinaryDiffEq.solve`.

Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.

!!! note
    This function only works when
    [OrdinaryDiffEq.jl](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl) is loaded with
    ```julia
    using OrdinaryDiffEq
    ```

!!! warning "Deprecated"
    This basis-based method is deprecated and will be removed in a future release.
    Consider using the chart-based variant [`solve_chart_exp_ode`](@ref Manifolds.solve_chart_exp_ode)`(M::AbstractManifold, a, Xc, A::AbstractAtlas, i)` instead.
"""
solve_exp_ode(M::AbstractManifold, p, X, t::Number; kwargs...)
