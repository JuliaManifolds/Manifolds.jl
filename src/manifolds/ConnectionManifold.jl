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
while, for a [`MetricManifold`](@ref) they are computed as (in Einstein summation convention)

````math
Γ^{l}_{ij} = g^{kl} Γ_{ijk},
````

where ``Γ_{ijk}`` are the Christoffel symbols of the first kind
(see [`christoffel_symbols_first`](@ref)), and ``g^{kl}`` is the inverse of the local
representation of the metric tensor. The dimensions of the resulting multi-dimensional array
are ordered ``(l,i,j)``.
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

The dimensions of the resulting multi-dimensional array are ordered ``(i,j,k,l)``.
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

"""
    gaussian_curvature(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = default_differential_backend())

Compute the Gaussian curvature of the manifold `M` at the point `p` using basis `B`.
This is equal to half of the scalar Ricci curvature, see [`ricci_curvature`](@ref).
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
        solver = AutoVern9(Rodas5()),
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
"""
solve_exp_ode(M::AbstractManifold, p, X, t::Number; kwargs...)
