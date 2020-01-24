@doc doc"""
    Metric

Abstract type for the pseudo-Riemannian metric tensor $g$, a family of smoothly
varying inner products on the tangent space. See [`inner`](@ref).
"""
abstract type Metric end

# piping syntax for decoration
if VERSION ≥ v"1.3"
    (metric::Metric)(M::Manifold) = MetricManifold(M, metric)
    (::Type{T})(M::Manifold) where {T<:Metric} = MetricManifold(M, T())
end

"""
    MetricManifold{M<:Manifold,G<:Metric} <: Manifold

Equip a [`Manifold`](@ref) explicitly with a [`Metric`](@ref) `G`.

For a Metric Manifold, by default, assumes, that you implement the linear form
from [`local_metric`](@ref) in order to evaluate the exponential map.

If the corresponding [`Metric`](@ref) `G` yields closed form formulae for e.g.
the exponential map and this is implemented directly (without solving the ode),
you can of course still implement that directly.

# Constructor
    MetricManifold(M, G)

Generate the [`Manifold`](@ref) `M` as a manifold with the [`Metric`](@ref) `G`.
"""
struct MetricManifold{M<:Manifold,G<:Metric} <: Manifold
    manifold::M
    metric::G
end

@doc doc"""
    LorentzMetric <: Metric

Abstract type for Lorentz metrics, which have a single time dimension. These
metrics assume the spacelike convention with the time dimension being last,
giving the signature $(++...+-)$.
"""
abstract type LorentzMetric <: Metric end

@doc doc"""
    RiemannianMetric <: Metric

Abstract type for Riemannian metrics, a family of positive definite inner
products. The positive definite property means that for $v \in T_x M$, the
inner product $g(v, v) > 0$ whenever $v$ is not the zero vector.
"""
abstract type RiemannianMetric <: Metric end

function check_manifold_point(M::MetricManifold, x; kwargs...)
    return check_manifold_point(M.manifold, x; kwargs...)
end

function check_tangent_vector(M::MetricManifold, x, v; kwargs...)
    return check_tangent_vector(M.manifold, x, v; kwargs...)
end

@doc doc"""
    christoffel_symbols_first(M::MetricManifold, x; backend=:default)

Compute the Christoffel symbols of the first kind in local coordinates.
The Christoffel symbols are (in Einstein summation convention)

$\Gamma_{ijk} = \frac{1}{2} \left[g_{kj,i} + g_{ik,j} - g_{ij,k}\right],$

where $g_{ij,k}=\frac{\partial}{\partial x^k} g_{ij}$ is the coordinate
derivative of the local representation of the metric tensor. The dimensions of
the resulting multi-dimensional array are ordered $(i,j,k)$.
"""
function christoffel_symbols_first(M::MetricManifold, x; backend = :default)
    ∂g = local_metric_jacobian(M, x; backend = backend)
    n = size(∂g, 1)
    Γ = similar(∂g, Size(n, n, n))
    @einsum Γ[i, j, k] = 1 / 2 * (∂g[k, j, i] + ∂g[i, k, j] - ∂g[i, j, k])
    return Γ
end

@doc doc"""
    christoffel_symbols_second(M::MetricManifold, x; backend=:default)

Compute the Christoffel symbols of the second kind in local coordinates.
The Christoffel symbols are (in Einstein summation convention)

$\Gamma^{l}_{ij} = g^{kl} \Gamma_{ijk},$

where $\Gamma_{ijk}$ are the Christoffel symbols of the first kind, and
$g^{kl}$ is the inverse of the local representation of the metric tensor.
The dimensions of the resulting multi-dimensional array are ordered $(l,i,j)$.
"""
function christoffel_symbols_second(M::MetricManifold, x; backend = :default)
    ginv = inverse_local_metric(M, x)
    Γ₁ = christoffel_symbols_first(M, x; backend = backend)
    Γ₂ = similar(Γ₁)
    @einsum Γ₂[l, i, j] = ginv[k, l] * Γ₁[i, j, k]
    return Γ₂
end

@doc doc"""
    christoffel_symbols_second_jacobian(M::MetricManifold, x; backend = :default)

Get partial derivatives of the Christoffel symbols of the second kind
for manifold `M` at `x` with respect to the coordinates of `x`,
$\frac{\partial}{\partial x^l} \Gamma^{k}_{ij} = \Gamma^{k}_{ij,l}.$
The dimensions of the resulting multi-dimensional array are ordered $(i,j,k,l)$.
"""
function christoffel_symbols_second_jacobian(M::MetricManifold, x; backend = :default)
    n = size(x, 1)
    ∂Γ = reshape(
        _jacobian(x -> christoffel_symbols_second(M, x; backend = backend), x, backend),
        n,
        n,
        n,
        n,
    )
    return ∂Γ
end

@doc doc"""
    det_local_metric(M::MetricManifold, x)

Return the determinant of local matrix representation of the metric tensor $g$.
"""
det_local_metric(M::MetricManifold, x) = det(local_metric(M, x))

"""
    einstein_tensor(M::MetricManifold, x; backend = :default)

Compute the Einstein tensor of the manifold `M` at the point `x`.
"""
function einstein_tensor(M::MetricManifold, x; backend = :default)
    Ric = ricci_tensor(M, x; backend = backend)
    g = local_metric(M, x)
    ginv = inverse_local_metric(M, x)
    S = sum(ginv .* Ric)
    G = Ric - g .* S / 2
    return G
end

@doc doc"""
    exp(N::MetricManifold{M,G}, x, v)

Copute the exponential map on the [`Manifold`](@ref) `M` equipped with the [`Metric`](@ref) `G`.

If the metric was declared the default metric using [`is_default_metric`](@ref), this method
falls back to `exp(M,x,v)`.

Otherwise it numerically integrates the underlying ODE, see [`solve_exp_ode`](@ref).
Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.
"""
exp(::MetricManifold, ::Any...)
function exp(M::MMT, x, v, T::AbstractVector{T} where {T}) where {MMT<:MetricManifold}
    return exp(M, is_default_metric(M), x, v, T)
end
function exp(
    M::MMT,
    ::Val{true},
    x,
    v,
    T::AbstractVector{T} where {T},
) where {MMT<:MetricManifold}
    return exp(base_manifold(M), x, v, T)
end
function exp(
    M::MMT,
    ::Val{false},
    x,
    v,
    T::AbstractVector{T} where {T},
) where {MMT<:MetricManifold}
    sol = solve_exp_ode(M, x, v, extrema(T); dense = false, saveat = T)
    n = length(x)
    return map(i -> sol.u[i][n+1:end], 1:length(T))
end

exp!(M::MMT, y, x, v) where {MMT<:MetricManifold} = exp!(M, is_default_metric(M), y, x, v)
function exp!(M::MMT, ::Val{true}, y, x, v) where {MMT<:MetricManifold}
    return exp!(base_manifold(M), y, x, v)
end
function exp!(M::MMT, ::Val{false}, y, x, v) where {MMT<:MetricManifold}
    tspan = (0.0, 1.0)
    sol = solve_exp_ode(M, x, v, tspan; dense = false, saveat = [1.0])
    n = length(x)
    return copyto!(y, sol.u[1][n+1:end])
end

@doc doc"""
    flat(N::MetricManifold{M,G}, x, w::FVector{TangentSpaceType})

Compute the musical isomorphism to transform the tangent vector `w` from the
[`Manifold`](@ref) `M` equipped with [`Metric`](@ref) `G` to a cotangent by
computing

````math
w^\flat = G_xw,
````
where $G_x$ is the local matrix representation of `G`, see [`local_metric`](@ref)
"""
flat(::MetricManifold, ::Any...)

function flat!(M::MMT, v::CoTFVector, x, w::TFVector) where {MMT<:MetricManifold}
    g = local_metric(M, x)
    copyto!(v.data, g * w.data)
    return v
end

"""
    gaussian_curvature(M::MetricManifold, x; backend = :default)

Compute the Gaussian curvature of the manifold `M` at the point `x`.
"""
gaussian_curvature(M::MetricManifold, x; kwargs...) = ricci_curvature(M, x; kwargs...) / 2

function get_basis(M::MMT, x, B::ArbitraryOrthonormalBasis) where {MMT<:MetricManifold}
    return invoke(get_basis, Tuple{MMT,Any,AbstractBasis}, M, x, B)
end
function get_basis(M::MMT, x, B::AbstractBasis) where {MMT<:MetricManifold}
    return get_basis(M, is_default_metric(M), x, B)
end
function get_basis(M::MMT, ::Val{true}, x, B::AbstractBasis) where {MMT<:MetricManifold}
    return get_basis(base_manifold(M), x, B)
end
function get_basis(M::MMT, ::Val{false}, x, B::AbstractBasis) where {MMT<:MetricManifold}
    error("tangent_orthogonal_basis not implemented on $(typeof(M)) for point $(typeof(x)) and basis type $(typeof(B)).")
end

function injectivity_radius(M::MMT, args...) where {MMT<:MetricManifold}
    return injectivity_radius(base_manifold(M), args...)
end

@doc doc"""
    inverse_local_metric(M::MetricManifold, x)

Return the local matrix representation of the inverse metric (cometric) tensor, usually
written $g^{ij}$.
"""
inverse_local_metric(M::MetricManifold, x) = inv(local_metric(M, x))

is_decorator_manifold(M::MMT) where {MMT<:MetricManifold} = Val(true)

"""
    is_default_metric(M,G)

Indicate whether the [`Metric`](@ref) `G` is the default metric for
the [`Manifold`](@ref) `M`. This means that any occurence of
[`MetricManifold`](@ref)(M,G) where `typeof(is_default_metric(M,G)) = Val{true}`
falls back to just be called with `M` such that the [`Manifold`](@ref) `M`
implicitly has this metric, for example if this was the first one implemented
or is the one most commonly assumed to be used.
"""
is_default_metric(M::Manifold, G::Metric) = Val(false)

"""
    is_default_metric(MM)

Indicate whether the [`Metric`](@ref) `MM.G` is the default metric for
the [`Manifold`](@ref) `MM.M` within the [`MetricManifold`](@ref) `MM`.
This means that any occurence of
[`MetricManifold`](@ref)`(MM.M,MM.G)` where `typeof(is_default_metric(MM.M,MM.G)) = Val{true}`
falls back to just be called with `MM.MM` such that the [`Manifold`](@ref) `MM.M`
implicitly has the metric `MM.G`, for example if this was the first one
implemented or is the one most commonly assumed to be used.
"""
function is_default_metric(M::MMT) where {MMT<:MetricManifold}
    return is_default_metric(base_manifold(M), metric(M))
end

function convert(T::Type{MetricManifold{MT,GT}}, M::MT) where {MT,GT}
    return _convert_with_default(M, GT, is_default_metric(M, GT()))
end

function _convert_with_default(M::MT, T::Type{<:Metric}, ::Val{true}) where {MT<:Manifold}
    return MetricManifold(M, T())
end
function _convert_with_default(M::MT, T::Type{<:Metric}, ::Val{false}) where {MT<:Manifold}
    error("Can not convert $(M) to a MetricManifold{$(MT),$(T)}, since $(T) is not the default metric.")
end

@doc doc"""
    inner(N::MetricManifold{M,G}, x, v, w)

Compute the inner product of `v`, `w` from the tangent space at `x` on the
[`Manifold`](@ref) `M` using the [`Metric`](@ref) `G`. If `G` is the default
metric (see [`is_default_metric`](@ref)) this is done using `log(M, x, v, w)`,
otherwise the [`local_metric`](@ref)`(M, x)` is employed as

````math
g_x(v,w) = \langle v, G_x w\rangle,
````
where $G_x$ is the local matrix representation of the [`Metric`](@ref) `G`.
"""
inner(M::MMT, x, v, w) where {MMT<:MetricManifold} = inner(M, is_default_metric(M), x, v, w)
function inner(M::MMT, ::Val{false}, x, v, w) where {MMT<:MetricManifold}
    return dot(v, local_metric(M, x) * w)
end
function inner(M::MMT, ::Val{true}, x, v, w) where {MMT<:MetricManifold}
    return inner(base_manifold(M), x, v, w)
end
function inner(
    B::VectorBundleFibers{<:CotangentSpaceType,MMT},
    x,
    v,
    w,
) where {MMT<:MetricManifold}
    ginv = inverse_local_metric(B.M, x)
    return dot(v, ginv * w)
end

@doc doc"""
    local_metric(M::MetricManifold, x)

Return the local matrix representation at the point `x` of the metric tensor $g$ on the
[`Manifold`](@ref) `M`, usually written $g_{ij}$. The matrix has the property that
$g(v, w)=v^T [g_{ij}] w = g_{ij} v^i w^j$, where the latter expression uses
Einstein summation convention.
"""
function local_metric(M::MetricManifold, x)
    error("Local metric not implemented on $(typeof(M)) for point $(typeof(x))")
end

@doc doc"""
    local_metric_jacobian(M::MetricManifold, x; backend=:default)

Get partial derivatives of the local metric of `M` at `x` with respect to the
coordinates of `x`, $\frac{\partial}{\partial x^k} g_{ij} = g_{ij,k}$. The
dimensions of the resulting multi-dimensional array are ordered $(i,j,k)$.
"""
function local_metric_jacobian(M, x; backend = :default)
    n = size(x, 1)
    ∂g = reshape(_jacobian(x -> local_metric(M, x), x, backend), n, n, n)
    return ∂g
end

log!(M::MMT, w, x, y) where {MMT<:MetricManifold} = log!(M, is_default_metric(M), w, x, y)
function log!(M::MMT, ::Val{true}, w, x, y) where {MMT<:MetricManifold}
    return log!(base_manifold(M), w, x, y)
end
function log!(M::MMT, ::Val{false}, w, x, y) where {MMT<:MetricManifold}
    error("Logarithmic map not implemented on $(typeof(M)) for points $(typeof(x)) and $(typeof(y)).")
end

@doc doc"""
    log_local_metric_density(M::MetricManifold, x)

Return the natural logarithm of the metric density $\rho$ of `M` at `x`, which
is given by $\rho=\log \sqrt{|\det [g_{ij}]|}$.
"""
log_local_metric_density(M::MetricManifold, x) = log(abs(det_local_metric(M, x))) / 2

function mean!(
    M::MMT,
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return mean!(M, is_default_metric(M), y, x, w; kwargs...)
end
function mean!(
    M::MMT,
    ::Val{true},
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return mean!(base_manifold(M), y, x, w; kwargs...)
end
function mean!(
    M::MMT,
    ::Val{false},
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return mean!(M, y, x, w, GradientDescentEstimation(); kwargs...)
end

function median!(
    M::MMT,
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return median!(M, is_default_metric(M), y, x, w; kwargs...)
end
function median!(
    M::MMT,
    ::Val{true},
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return median!(base_manifold(M), y, x, w; kwargs...)
end
function median!(
    M::MMT,
    ::Val{false},
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return median!(M, y, x, w, CyclicProximalPointEstimation(); kwargs...)
end

@doc doc"""
    metric(M::MetricManifold)

Get the metric $g$ of the manifold `M`.
"""
metric(M::MetricManifold) = M.metric

function normal_tvector_distribution(M::MMT, x, σ) where {MMT<:MetricManifold}
    return normal_tvector_distribution(M, is_default_metric(M), x, σ)
end
function normal_tvector_distribution(M::MMT, ::Val{true}, x, σ) where {MMT<:MetricManifold}
    return normal_tvector_distribution(base_manifold(M), x, σ)
end
function normal_tvector_distribution(M::MMT, ::Val{false}, x, σ) where {MMT<:MetricManifold}
    error("normal_tvector_distribution not implemented for a $(typeof(M)) at point $(typeof(x)) with standard deviation $(typeof(σ)).")
end

function project_point!(M::MMT, y, x) where {MMT<:MetricManifold}
    return project_point!(M, is_default_metric(M), y, x)
end
function project_point!(M::MMT, ::Val{true}, y, x) where {MMT<:MetricManifold}
    return project_point!(base_manifold(M), y, x)
end
function project_point!(M::MMT, ::Val{false}, y, x) where {MMT<:MetricManifold}
    error("project_point! not implemented on $(typeof(M)) for point $(typeof(x))")
end

function project_tangent!(M::MMT, w, x, v) where {MMT<:MetricManifold}
    return project_tangent!(M, is_default_metric(M), w, x, v)
end
function project_tangent!(M::MMT, ::Val{true}, w, x, v) where {MMT<:MetricManifold}
    return project_tangent!(base_manifold(M), w, x, v)
end
function project_tangent!(M::MMT, ::Val{false}, w, x, v) where {MMT<:MetricManifold}
    error("project_tangent! not implemented for a $(typeof(M)) and tangent $(typeof(v)) at point $(typeof(x)).")
end

function projected_distribution(M::MMT, d, x) where {MMT<:MetricManifold}
    return projected_distribution(M, is_default_metric(M), d, x)
end
function projected_distribution(M::MMT, ::Val{true}, d, x) where {MMT<:MetricManifold}
    return projected_distribution(base_manifold(M), d, x)
end
function projected_distribution(M::MMT, ::Val{false}, d, x) where {MMT<:MetricManifold}
    error("projected_distribution not implemented for a $(typeof(M)) and with $(typeof(d)) at point $(typeof(x)).")
end
function projected_distribution(M::MMT, d) where {MMT<:MetricManifold}
    return projected_distribution(M, is_default_metric(M), d)
end
function projected_distribution(M::MMT, ::Val{true}, d) where {MMT<:MetricManifold}
    return projected_distribution(base_manifold(M), d)
end
function projected_distribution(M::MMT, ::Val{false}, d) where {MMT<:MetricManifold}
    error("projected_distribution not implemented for a $(typeof(M)) with $(typeof(d)).")
end

"""
    ricci_curvature(M::MetricManifold, x; backend = :default)

Compute the Ricci scalar curvature of the manifold `M` at the point `x`.
"""
function ricci_curvature(M::MetricManifold, x; backend = :default)
    ginv = inverse_local_metric(M, x)
    Ric = ricci_tensor(M, x; backend = backend)
    S = sum(ginv .* Ric)
    return S
end

"""
    ricci_tensor(M::MetricManifold, x; backend = :default)

Compute the Ricci tensor, also known as the Ricci curvature tensor,
of the manifold `M` at the point `x`.
"""
function ricci_tensor(M::MetricManifold, x; kwargs...)
    R = riemann_tensor(M, x; kwargs...)
    n = size(R, 1)
    Ric = similar(R, Size(n, n))
    @einsum Ric[i, j] = R[l, i, l, j]
    return Ric
end

@doc doc"""
    riemann_tensor(M::MetricManifold, x)

Compute the Riemann tensor $R^l_{ijk}$, also known as the Riemann curvature
tensor, at the point `x`. The dimensions of the resulting multi-dimensional
array are ordered $(l,i,j,k)$.
"""
function riemann_tensor(M::MetricManifold, x; backend = :default)
    n = size(x, 1)
    Γ = christoffel_symbols_second(M, x; backend = backend)
    ∂Γ = christoffel_symbols_second_jacobian(M, x; backend = backend) ./ n
    R = similar(∂Γ, Size(n, n, n, n))
    @einsum R[l, i, j, k] =
        ∂Γ[l, i, k, j] - ∂Γ[l, i, j, k] + Γ[s, i, k] * Γ[l, s, j] - Γ[s, i, j] * Γ[l, s, k]
    return R
end

@doc doc"""
    sharp(N::MetricManifold{M,G}, x, w::FVector{CotangentSpaceType})

Compute the musical isomorphism to transform the cotangent vector `w` from the
[`Manifold`](@ref) `M` equipped with [`Metric`](@ref) `G` to a tangent by
computing

````math
w^\sharp = G_x^{-1}w,
````
where $G_x$ is the local matrix representation of `G`, i.e. one employs
[`inverse_local_metric`](@ref) here to obtain $G_x^{-1}$.
"""
sharp(::MetricManifold, ::Any)

function sharp!(M::N, v::TFVector, x, w::CoTFVector) where {N<:MetricManifold}
    ginv = inverse_local_metric(M, x)
    copyto!(v.data, ginv * w.data)
    return v
end

@doc doc"""
    solve_exp_ode(
        M::MetricManifold,
        x,
        v,
        tspan;
        backend = :default,
        solver = AutoVern9(Rodas5()),
        kwargs...,
    )

Approximate the exponential map on the manifold over the provided timespan
assuming the Levi-Civita connection by solving the ordinary differential
equation

$\frac{d^2}{dt^2} x^k + \Gamma^k_{ij} \frac{d}{dt} x_i \frac{d}{dt} x_j = 0,$

where $\Gamma^k_{ij}$ are the Christoffel symbols of the second kind, and
the Einstein summation convention is assumed. The arguments `tspan` and
`solver` follow the `OrdinaryDiffEq` conventions. `kwargs...` specify keyword
arguments that will be passed to `OrdinaryDiffEq.solve`.

Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.

!!! note
    This function only works for Julia 1.1 or greater, when
    [OrdinaryDiffEq.jl](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl) is loaded with
    ```julia
    using OrdinaryDiffEq
    ```
"""
function solve_exp_ode(M, x, v, tspan; kwargs...)
    error("solve_exp_ode not implemented on $(typeof(M)) for point $(typeof(x)), vector $(typeof(v)), and timespan $(typeof(tspan)). For a suitable default, enter `using OrdinaryDiffEq` on Julia 1.1 or greater.")
end

function vector_transport_to!(
    M::MMT,
    vto,
    x,
    v,
    y,
    m::AbstractVectorTransportMethod,
) where {MMT<:MetricManifold}
    return vector_transport_to!(M, is_default_metric(M), vto, x, v, y, m)
end
function vector_transport_to!(
    M::MMT,
    ::Val{true},
    vto,
    x,
    v,
    y,
    m,
) where {MMT<:MetricManifold}
    return vector_transport_to!(base_manifold(M), vto, x, v, y, m)
end
function vector_transport_to!(
    M::MMT,
    ::Val{false},
    vto,
    x,
    v,
    y,
    m,
) where {MMT<:MetricManifold}
    error("vector transport from a point of type $(typeof(x)) to a type $(typeof(y)) on a $(typeof(M)) for a vector of type $(v) and the $(typeof(m)) not yet implemented.")
end

function zero_tangent_vector!(M::MMT, v, x) where {MMT<:MetricManifold}
    return zero_tangent_vector!(M.manifold, v, x)
end
