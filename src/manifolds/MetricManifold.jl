@doc raw"""
    Metric

Abstract type for the pseudo-Riemannian metric tensor $g$, a family of smoothly
varying inner products on the tangent space. See [`inner`](@ref).
"""
abstract type Metric end

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

@doc raw"""
    LorentzMetric <: Metric

Abstract type for Lorentz metrics, which have a single time dimension. These
metrics assume the spacelike convention with the time dimension being last,
giving the signature $(++...+-)$.
"""
abstract type LorentzMetric <: Metric end

@doc raw"""
    RiemannianMetric <: Metric

Abstract type for Riemannian metrics, a family of positive definite inner
products. The positive definite property means that for $X  ∈ T_p \mathcal M$, the
inner product $g(X, X) > 0$ whenever $X$ is not the zero vector.
"""
abstract type RiemannianMetric <: Metric end

function check_manifold_point(M::MetricManifold, p; kwargs...)
    return check_manifold_point(M.manifold, p; kwargs...)
end

function check_tangent_vector(M::MetricManifold, p, X; kwargs...)
    return check_tangent_vector(M.manifold, p, X; kwargs...)
end

@doc raw"""
    christoffel_symbols_first(M::MetricManifold, p; backend=:default)

Compute the Christoffel symbols of the first kind in local coordinates.
The Christoffel symbols are (in Einstein summation convention)

$\Gamma_{ijk} = \frac{1}{2} \Bigl[g_{kj,i} + g_{ik,j} - g_{ij,k}\Bigr],$

where $g_{ij,k}=\frac{∂}{∂ p^k} g_{ij}$ is the coordinate
derivative of the local representation of the metric tensor. The dimensions of
the resulting multi-dimensional array are ordered $(i,j,k)$.
"""
function christoffel_symbols_first(M::MetricManifold, p; backend = :default)
    ∂g = local_metric_jacobian(M, p; backend = backend)
    n = size(∂g, 1)
    Γ = allocate(∂g, Size(n, n, n))
    @einsum Γ[i, j, k] = 1 / 2 * (∂g[k, j, i] + ∂g[i, k, j] - ∂g[i, j, k])
    return Γ
end

@doc raw"""
    christoffel_symbols_second(M::MetricManifold, x; backend=:default)

Compute the Christoffel symbols of the second kind in local coordinates.
The Christoffel symbols are (in Einstein summation convention)

$\Gamma^{l}_{ij} = g^{kl} \Gamma_{ijk},$

where $\Gamma_{ijk}$ are the Christoffel symbols of the first kind, and
$g^{kl}$ is the inverse of the local representation of the metric tensor.
The dimensions of the resulting multi-dimensional array are ordered $(l,i,j)$.
"""
function christoffel_symbols_second(M::MetricManifold, p; backend = :default)
    ginv = inverse_local_metric(M, p)
    Γ₁ = christoffel_symbols_first(M, p; backend = backend)
    Γ₂ = allocate(Γ₁)
    @einsum Γ₂[l, i, j] = ginv[k, l] * Γ₁[i, j, k]
    return Γ₂
end

@doc raw"""
    christoffel_symbols_second_jacobian(M::MetricManifold, p; backend = :default)

Get partial derivatives of the Christoffel symbols of the second kind
for manifold `M` at `p` with respect to the coordinates of `p`,
$\frac{∂}{∂ p^l} \Gamma^{k}_{ij} = \Gamma^{k}_{ij,l}.$
The dimensions of the resulting multi-dimensional array are ordered $(i,j,k,l)$.
"""
function christoffel_symbols_second_jacobian(M::MetricManifold, p; backend = :default)
    n = size(p, 1)
    ∂Γ = reshape(
        _jacobian(q -> christoffel_symbols_second(M, q; backend = backend), p, backend),
        n,
        n,
        n,
        n,
    )
    return ∂Γ
end

@doc raw"""
    det_local_metric(M::MetricManifold, p)

Return the determinant of local matrix representation of the metric tensor $g$.
"""
det_local_metric(M::MetricManifold, p) = det(local_metric(M, p))

"""
    einstein_tensor(M::MetricManifold, p; backend = :default)

Compute the Einstein tensor of the manifold `M` at the point `p`.
"""
function einstein_tensor(M::MetricManifold, p; backend = :default)
    Ric = ricci_tensor(M, p; backend = backend)
    g = local_metric(M, p)
    ginv = inverse_local_metric(M, p)
    S = sum(ginv .* Ric)
    G = Ric - g .* S / 2
    return G
end

@doc raw"""
    exp(N::MetricManifold{M,G}, p, X)

Copute the exponential map on the [`Manifold`](@ref) `M` equipped with the [`Metric`](@ref) `G`.

If the metric was declared the default metric using [`is_default_metric`](@ref), this method
falls back to `exp(M,p,X)`.

Otherwise it numerically integrates the underlying ODE, see [`solve_exp_ode`](@ref).
Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.
"""
exp(::MetricManifold, ::Any...)
function exp(M::MMT, p, X, T::AbstractVector{T} where {T}) where {MMT<:MetricManifold}
    return exp(M, is_default_metric(M), p, X, T)
end
function exp(
    M::MMT,
    ::Val{true},
    p,
    X,
    T::AbstractVector{T} where {T},
) where {MMT<:MetricManifold}
    return exp(base_manifold(M), p, X, T)
end
function exp(
    M::MMT,
    ::Val{false},
    p,
    X,
    T::AbstractVector{T} where {T},
) where {MMT<:MetricManifold}
    sol = solve_exp_ode(M, p, X, extrema(T); dense = false, saveat = T)
    n = length(p)
    return map(i -> sol.u[i][n+1:end], 1:length(T))
end

exp!(M::MMT, q, p, X) where {MMT<:MetricManifold} = exp!(M, is_default_metric(M), q, p, X)
function exp!(M::MMT, ::Val{true}, q, p, X) where {MMT<:MetricManifold}
    return exp!(base_manifold(M), q, p, X)
end
function exp!(M::MMT, ::Val{false}, q, p, X) where {MMT<:MetricManifold}
    tspan = (0.0, 1.0)
    sol = solve_exp_ode(M, p, X, tspan; dense = false, saveat = [1.0])
    n = length(p)
    return copyto!(q, sol.u[1][n+1:end])
end

@doc raw"""
    flat(N::MetricManifold{M,G}, p, X::FVector{TangentSpaceType})

Compute the musical isomorphism to transform the tangent vector `X` from the
[`Manifold`](@ref) `M` equipped with [`Metric`](@ref) `G` to a cotangent by
computing

````math
X^\flat = G_pX,
````
where $G_p$ is the local matrix representation of `G`, see [`local_metric`](@ref)
"""
flat(::MetricManifold, ::Any...)

function flat!(M::MMT, ξ::CoTFVector, p, X::TFVector) where {MMT<:MetricManifold}
    g = local_metric(M, p)
    copyto!(ξ.data, g * X.data)
    return ξ
end

"""
    gaussian_curvature(M::MetricManifold, x; backend = :default)

Compute the Gaussian curvature of the manifold `M` at the point `x`.
"""
gaussian_curvature(M::MetricManifold, p; kwargs...) = ricci_curvature(M, p; kwargs...) / 2

function get_basis(M::MMT, p, B::ArbitraryOrthonormalBasis) where {MMT<:MetricManifold}
    return invoke(get_basis, Tuple{MMT,Any,AbstractBasis}, M, p, B)
end
function get_basis(M::MMT, p, B::AbstractBasis) where {MMT<:MetricManifold}
    return get_basis(M, is_default_metric(M), p, B)
end
function get_basis(M::MMT, ::Val{true}, p, B::AbstractBasis) where {MMT<:MetricManifold}
    return get_basis(base_manifold(M), p, B)
end
function get_basis(M::MMT, ::Val{false}, p, B::AbstractBasis) where {MMT<:MetricManifold}
    error("tangent_orthogonal_basis not implemented on $(typeof(M)) for point $(typeof(p)) and basis type $(typeof(B)).")
end

function injectivity_radius(M::MMT, args...) where {MMT<:MetricManifold}
    return injectivity_radius(base_manifold(M), args...)
end

@doc raw"""
    inverse_local_metric(M::MetricManifold, p)

Return the local matrix representation of the inverse metric (cometric) tensor, usually
written $g^{ij}$.
"""
inverse_local_metric(M::MetricManifold, p) = inv(local_metric(M, p))

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
the [`Manifold`](@ref) `MM.manifold,` within the [`MetricManifold`](@ref) `MM`.
This means that any occurence of
[`MetricManifold`](@ref)`(MM.manifold,MM.G)` where `typeof(is_default_metric(MM.manifold,MM.G)) = Val{true}`
falls back to just be called with `MM.manifold,` such that the [`Manifold`](@ref) `MM.manifold`
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

@doc raw"""
    inner(N::MetricManifold{M,G}, p, X, Y)

Compute the inner product of `X` and `Y` from the tangent space at `p` on the
[`Manifold`](@ref) `M` using the [`Metric`](@ref) `G`. If `G` is the default
metric (see [`is_default_metric`](@ref)) this is done using `inner(M, p, X, Y)`,
otherwise the [`local_metric`](@ref)`(M, p)` is employed as

````math
g_p(X, Y) = ⟨X, G_pY⟩,
````
where $G_p$ is the local matrix representation of the [`Metric`](@ref) `G`.
"""
inner(M::MMT, p, X, Y) where {MMT<:MetricManifold} = inner(M, is_default_metric(M), p, X, Y)
function inner(M::MMT, ::Val{false}, p, X, Y) where {MMT<:MetricManifold}
    return dot(X, local_metric(M, p) * Y)
end
function inner(M::MMT, ::Val{true}, p, X, Y) where {MMT<:MetricManifold}
    return inner(base_manifold(M), p, X, Y)
end
function inner(
    B::VectorBundleFibers{<:CotangentSpaceType,MMT},
    p,
    X,
    Y,
) where {MMT<:MetricManifold}
    ginv = inverse_local_metric(B.manifold, p)
    return dot(X, ginv * Y)
end

@doc raw"""
    local_metric(M::MetricManifold, p)

Return the local matrix representation at the point `p` of the metric
tensor $g$ on the [`Manifold`](@ref) `M`, usually written $g_{ij}$.
The matrix has the property that $g(v, w)=v^T [g_{ij}] w = g_{ij} v^i w^j$,
where the latter expression uses Einstein summation convention.
"""
function local_metric(M::MetricManifold, p)
    error("Local metric not implemented on $(typeof(M)) for point $(typeof(p))")
end

@doc raw"""
    local_metric_jacobian(M::MetricManifold, p; backend=:default)

Get partial derivatives of the local metric of `M` at `p` with respect to the
coordinates of `p`, $\frac{∂}{∂ p^k} g_{ij} = g_{ij,k}$. The
dimensions of the resulting multi-dimensional array are ordered $(i,j,k)$.
"""
function local_metric_jacobian(M, p; backend = :default)
    n = size(p, 1)
    ∂g = reshape(_jacobian(q -> local_metric(M, q), p, backend), n, n, n)
    return ∂g
end

log!(M::MMT, w, x, y) where {MMT<:MetricManifold} = log!(M, is_default_metric(M), w, x, y)
function log!(M::MMT, ::Val{true}, w, x, y) where {MMT<:MetricManifold}
    return log!(base_manifold(M), w, x, y)
end
function log!(M::MMT, ::Val{false}, Y, p, q) where {MMT<:MetricManifold}
    error("Logarithmic map not implemented on $(typeof(M)) for points $(typeof(p)) and $(typeof(q)).")
end

@doc raw"""
    log_local_metric_density(M::MetricManifold, p)

Return the natural logarithm of the metric density $\rho$ of `M` at `p`, which
is given by $\rho=\log \sqrt{|\det [g_{ij}]|}$.
"""
log_local_metric_density(M::MetricManifold, p) = log(abs(det_local_metric(M, p))) / 2

function mean!(
    M::MMT,
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return mean!(M, is_default_metric(M), p, x, w; kwargs...)
end
function mean!(
    M::MMT,
    ::Val{true},
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return mean!(base_manifold(M), p, x, w; kwargs...)
end
function mean!(
    M::MMT,
    ::Val{false},
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return mean!(M, p, x, w, GradientDescentEstimation(); kwargs...)
end

function median!(
    M::MMT,
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return median!(M, is_default_metric(M), p, x, w; kwargs...)
end
function median!(
    M::MMT,
    ::Val{true},
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return median!(base_manifold(M), p, x, w; kwargs...)
end
function median!(
    M::MMT,
    ::Val{false},
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {MMT<:MetricManifold}
    return median!(M, p, x, w, CyclicProximalPointEstimation(); kwargs...)
end

@doc raw"""
    metric(M::MetricManifold)

Get the metric $g$ of the manifold `M`.
"""
metric(M::MetricManifold) = M.metric

function normal_tvector_distribution(M::MMT, p, σ) where {MMT<:MetricManifold}
    return normal_tvector_distribution(M, is_default_metric(M), p, σ)
end
function normal_tvector_distribution(M::MMT, ::Val{true}, p, σ) where {MMT<:MetricManifold}
    return normal_tvector_distribution(base_manifold(M), p, σ)
end
function normal_tvector_distribution(M::MMT, ::Val{false}, p, σ) where {MMT<:MetricManifold}
    error("normal_tvector_distribution not implemented for a $(typeof(M)) at point $(typeof(p)) with standard deviation $(typeof(σ)).")
end

function project_point!(M::MMT, q, p) where {MMT<:MetricManifold}
    return project_point!(M, is_default_metric(M), q, p)
end
function project_point!(M::MMT, ::Val{true}, q, p) where {MMT<:MetricManifold}
    return project_point!(base_manifold(M), q, p)
end
function project_point!(M::MMT, ::Val{false}, q, p) where {MMT<:MetricManifold}
    error("project_point! not implemented on $(typeof(M)) for point $(typeof(p))")
end

function project_tangent!(M::MMT, Y, p, X) where {MMT<:MetricManifold}
    return project_tangent!(M, is_default_metric(M), Y, p, X)
end
function project_tangent!(M::MMT, ::Val{true}, Y, p, X) where {MMT<:MetricManifold}
    return project_tangent!(base_manifold(M), Y, p, X)
end
function project_tangent!(M::MMT, ::Val{false}, Y, p, X) where {MMT<:MetricManifold}
    error("project_tangent! not implemented for a $(typeof(M)) and tangent $(typeof(X)) at point $(typeof(p)).")
end

function projected_distribution(M::MMT, d, p) where {MMT<:MetricManifold}
    return projected_distribution(M, is_default_metric(M), d, p)
end
function projected_distribution(M::MMT, ::Val{true}, d, p) where {MMT<:MetricManifold}
    return projected_distribution(base_manifold(M), d, p)
end
function projected_distribution(M::MMT, ::Val{false}, d, p) where {MMT<:MetricManifold}
    error("projected_distribution not implemented for a $(typeof(M)) and with $(typeof(d)) at point $(typeof(p)).")
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
    ricci_curvature(M::MetricManifold, p; backend = :default)

Compute the Ricci scalar curvature of the manifold `M` at the point `p`.
"""
function ricci_curvature(M::MetricManifold, p; backend = :default)
    ginv = inverse_local_metric(M, p)
    Ric = ricci_tensor(M, p; backend = backend)
    S = sum(ginv .* Ric)
    return S
end

"""
    ricci_tensor(M::MetricManifold, p; backend = :default)

Compute the Ricci tensor, also known as the Ricci curvature tensor,
of the manifold `M` at the point `p`.
"""
function ricci_tensor(M::MetricManifold, p; kwargs...)
    R = riemann_tensor(M, p; kwargs...)
    n = size(R, 1)
    Ric = allocate(R, Size(n, n))
    @einsum Ric[i, j] = R[l, i, l, j]
    return Ric
end

@doc raw"""
    riemann_tensor(M::MetricManifold, p)

Compute the Riemann tensor $R^l_{ijk}$, also known as the Riemann curvature
tensor, at the point `p`. The dimensions of the resulting multi-dimensional
array are ordered $(l,i,j,k)$.
"""
function riemann_tensor(M::MetricManifold, p; backend = :default)
    n = size(p, 1)
    Γ = christoffel_symbols_second(M, p; backend = backend)
    ∂Γ = christoffel_symbols_second_jacobian(M, p; backend = backend) ./ n
    R = allocate(∂Γ, Size(n, n, n, n))
    @einsum R[l, i, j, k] =
        ∂Γ[l, i, k, j] - ∂Γ[l, i, j, k] + Γ[s, i, k] * Γ[l, s, j] - Γ[s, i, j] * Γ[l, s, k]
    return R
end

@doc raw"""
    sharp(N::MetricManifold{M,G}, p, ξ::FVector{CotangentSpaceType})

Compute the musical isomorphism to transform the cotangent vector `ξ` from the
[`Manifold`](@ref) `M` equipped with [`Metric`](@ref) `G` to a tangent by
computing

````math
ξ^\sharp = G_p^{-1}ξ,
````
where $G_p$ is the local matrix representation of `G`, i.e. one employs
[`inverse_local_metric`](@ref) here to obtain $G_p^{-1}$.
"""
sharp(::MetricManifold, ::Any)

function sharp!(M::N, X::TFVector, p, ξ::CoTFVector) where {N<:MetricManifold}
    ginv = inverse_local_metric(M, p)
    copyto!(X.data, ginv * ξ.data)
    return X
end

show(io::IO, M::MetricManifold) = print(io, "MetricManifold($(M.manifold), $(M.metric))")

@doc raw"""
    solve_exp_ode(
        M::MetricManifold,
        p,
        X,
        tspan;
        backend = :default,
        solver = AutoVern9(Rodas5()),
        kwargs...,
    )

Approximate the exponential map on the manifold over the provided timespan
assuming the Levi-Civita connection by solving the ordinary differential
equation

$\frac{d^2}{dt^2} p^k + \Gamma^k_{ij} \frac{d}{dt} p_i \frac{d}{dt} p_j = 0,$

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
function solve_exp_ode(M, p, X, tspan; kwargs...)
    error("solve_exp_ode not implemented on $(typeof(M)) for point $(typeof(p)), vector $(typeof(X)), and timespan $(typeof(tspan)). For a suitable default, enter `using OrdinaryDiffEq` on Julia 1.1 or greater.")
end

function vector_transport_to!(
    M::MMT,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
) where {MMT<:MetricManifold}
    return vector_transport_to!(M, is_default_metric(M), Y, p, X, q, m)
end
function vector_transport_to!(
    M::MMT,
    ::Val{true},
    Y,
    p,
    X,
    q,
    m,
) where {MMT<:MetricManifold}
    return vector_transport_to!(base_manifold(M), Y, p, X, q, m)
end
function vector_transport_to!(
    M::MMT,
    ::Val{false},
    Y,
    p,
    X,
    q,
    m,
) where {MMT<:MetricManifold}
    error("vector transport from a point of type $(typeof(p)) to a type $(typeof(q)) on a $(typeof(M)) for a vector of type $(X) and the $(typeof(m)) not yet implemented.")
end

function zero_tangent_vector!(M::MMT, X, p) where {MMT<:MetricManifold}
    return zero_tangent_vector!(M.manifold, X, p)
end
