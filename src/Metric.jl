@doc doc"""
    Metric

Abstract type for the pseudo-Riemannian metric tensor $g$, a family of smoothly
varying inner products on the tangent space. See [`inner`](@ref).
"""
abstract type Metric end

@doc doc"""
    RiemannianMetric <: Metric

Abstract type for Riemannian metrics, a family of positive definite inner
products. The positive definite property means that for $v \in T_x M$, the
inner product $g(v, v) > 0$ whenever $v$ is not the zero vector.
"""
abstract type RiemannianMetric <: Metric end

@doc doc"""
    LorentzMetric <: Metric

Abstract type for Lorentz metrics, which have a single time dimension. These
metrics assume the spacelike convention with the time dimension being last,
giving the signature $(++...+-)$.
"""
abstract type LorentzMetric <: Metric end

"""
    MetricManifold{M<:Manifold,G<:Metric} <: Manifold

Equip a [`Manifold`](@ref) explicitly with a [`Metric`](@ref) `G`. 

For a Metric Manifold, by default, assumes, that you implement the linear form
from [`local_metric`](@ref) in order to evaluate the exponential map.

If the corresponding [`Metric`](@ref) `G` yields closed form formulae for e.g.
the exponential map and this is implemented directly (without solving the ode),
you of course can still implement that directly.

# Constructor
    MetricManifold(manifold, metric)

    """
struct MetricManifold{M<:Manifold,G<:Metric} <: Manifold
    manifold::M
    metric::G
end

is_decorator_manifold(M::MMT) where {MMT <: MetricManifold}= Val(true)

"""
    is_default_metric(M,G)

indicate, whether the [`Metric`](@ref) `G` is the default metric for
the [`Manifold`](@ref) `M`. This means that any occurence of
[`MetricManifold`](@ref)(M,G) where `is_default_metric(M,G) = Val{true}`
falls back to just be called with `M` such that the [`Manifold`](@ref) `M`
implicitly has this metric, for example if this was the first one implemented
or is the one most commonly assumed to be used.
"""
is_default_metric(M::Manifold,G::Metric) = Val(false)
"""
    is_default_metric(MM)

indicate, whether the [`Metric`](@ref) `G` is the default metric for
the [`Manifold`](@ref) `M` within the [`MetricManifold`](@ref) `M`.
This means that any occurence of
[`MetricManifold`](@ref)`(M,G)` where `is_default_metric(M,G) = Val{true}`
falls back to just be called with `M` such that the [`Manifold`](@ref) `M`
implicitly has this metric, for example if this was the first one implemented
or is the one most commonly assumed to be used.
"""
is_default_metric(M::MMT) where {MMT <: MetricManifold} = is_default_metric(base_manifold(M),metric(M))
# this automatically undecorates
# this should automatically decorate at least for simple cases
convert(T::Type{MetricManifold{MT,GT}},M::MT) where {MT,GT} = _convert_with_default(M,GT,is_default_metric(M,GT()))
_convert_with_default(M::MT,T::Type{<:Metric},::Val{true}) where {MT <: Manifold} = MetricManifold(M,T())
_convert_with_default(M::MT,T::Type{<:Metric},::Val{false}) where {MT <: Manifold} = error("Can not convert $(M) to a MetricManifold{$(MT),$(T)}, since $(T) is not the default metric.")

@doc doc"""
    metric(M::MetricManifold)

Get the metric $g$ of the manifold `M`.
"""
metric(M::MetricManifold) = M.metric

@doc doc"""
    local_metric(M::MetricManifold, x)

Local matrix representation at the point `x` of the metric tensor $g$ on the
manifold `M`, usually written $g_{ij}$. The matrix has the property that
$g(v, w)=v^T [g_{ij}] w = g_{ij} v^i w^j$, where the latter expression uses
Einstein summation convention.
"""
function local_metric(M::MetricManifold, x)
    error("Local metric not implemented on $(typeof(M)) for point $(typeof(x))")
end

@doc doc"""
    inverse_local_metric(M::MetricManifold, x)

Local matrix representation of the inverse metric (cometric) tensor, usually
written $g^{ij}$
"""
inverse_local_metric(M::MetricManifold, x) = inv(local_metric(M, x))

@doc doc"""
    det_local_metric(M::MetricManifold, x)

Determinant of local matrix representation of the metric tensor $g$
"""
det_local_metric(M::MetricManifold, x) = det(local_metric(M, x))

@doc doc"""
    log_local_metric_density(M::MetricManifold, x)

Return the natural logarithm of the metric density $\rho$ of `M` at `x`, which
is given by $\rho=\log \sqrt{|\det [g_{ij}]|}$.
"""
log_local_metric_density(M::MetricManifold, x) = log(abs(det_local_metric(M, x))) / 2

@doc doc"""
    local_metric_jacobian(M::MetricManifold, x)

Get partial derivatives of the local metric of `M` at `x` with respect to the
coordinates of `x`, $\frac{\partial}{\partial x^k} g_{ij} = g_{ij,k}$. The
dimensions of the resulting multi-dimensional array are ordered $(i,j,k)$.
"""
function local_metric_jacobian(M, x)
    error("local_metric_jacobian not implemented on $(typeof(M)) for point $(typeof(x)). For a suitable default, enter `using ForwardDiff`.")
end

@doc doc"""
    christoffel_symbols_first(M::MetricManifold, x)

Compute the Christoffel symbols of the first kind in local coordinates.
The Christoffel symbols are (in Einstein summation convention)

$\Gamma_{ijk} = \frac{1}{2} \left[g_{kj,i} + g_{ik,j} - g_{ij,k}\right],$

where $g_{ij,k}=\frac{\partial}{\partial x^k} g_{ij}$ is the coordinate
derivative of the local representation of the metric tensor. The dimensions of
the resulting multi-dimensional array are ordered $(i,j,k)$.
"""
function christoffel_symbols_first(M::MetricManifold, x)
    ∂g = local_metric_jacobian(M, x)
    n = size(∂g, 1)
    Γ = similar(∂g, Size(n, n, n))
    @einsum Γ[i,j,k] = 1 / 2 * (∂g[k,j,i] + ∂g[i,k,j] - ∂g[i,j,k])
    return Γ
end

@doc doc"""
    christoffel_symbols_second(M::MetricManifold, x)

Compute the Christoffel symbols of the second kind in local coordinates.
The Christoffel symbols are (in Einstein summation convention)

$\Gamma^{l}_{ij} = g^{kl} \Gamma_{ijk},$

where $\Gamma_{ijk}$ are the Christoffel symbols of the first kind, and
$g^{kl}$ is the inverse of the local representation of the metric tensor.
The dimensions of the resulting multi-dimensional array are ordered $(l,i,j)$.
"""
function christoffel_symbols_second(M::MetricManifold, x)
    ginv = inverse_local_metric(M, x)
    Γ₁ = christoffel_symbols_first(M, x)
    Γ₂ = similar(Γ₁)
    @einsum Γ₂[l,i,j] = ginv[k,l] * Γ₁[i,j,k]
    return Γ₂
end

@doc doc"""
    christoffel_symbols_second_jacobian(M::MetricManifold, x)

Get partial derivatives of the Christoffel symbols of the second kind
for manifold `M` at `x` with respect to the coordinates of `x`,
$\frac{\partial}{\partial x^l} \Gamma^{k}_{ij} = \Gamma^{k}_{ij,l}.$
The dimensions of the resulting multi-dimensional array are ordered $(i,j,k,l)$.
"""
function christoffel_symbols_second_jacobian(M, x)
    error("christoffel_symbols_second_jacobian not implemented on $(typeof(M)) for point $(typeof(x)). For a suitable default, enter `using ForwardDiff`.")
end

@doc doc"""
    riemann_tensor(M::MetricManifold, x)

Compute the Riemann tensor $R^l_{ijk}$, also known as the Riemann curvature
tensor, at the point `x`. The dimensions of the resulting multi-dimensional
array are ordered $(l,i,j,k)$.
"""
function riemann_tensor(M::MetricManifold, x)
    n = size(x, 1)
    Γ = christoffel_symbols_second(M, x)
    ∂Γ = christoffel_symbols_second_jacobian(M, x) ./ n
    R = similar(∂Γ, Size(n, n, n, n))
    @einsum R[l,i,j,k] = ∂Γ[l,i,k,j] - ∂Γ[l,i,j,k] + Γ[s,i,k] * Γ[l,s,j] - Γ[s,i,j] * Γ[l,s,k]
    return R
end

"""
    ricci_tensor(M::MetricManifold, x)

Compute the Ricci tensor, also known as the Ricci curvature tensor,
of the manifold `M` at the point `x`.
"""
function ricci_tensor(M::MetricManifold, x)
    R = riemann_tensor(M, x)
    n = size(R, 1)
    Ric = similar(R, Size(n, n))
    @einsum Ric[i,j] = R[l,i,l,j]
    return Ric
end

"""
    ricci_curvature(M::MetricManifold, x)

Compute the Ricci scalar curvature of the manifold `M` at the point `x`.
"""
function ricci_curvature(M::MetricManifold, x)
    ginv = inverse_local_metric(M, x)
    Ric = ricci_tensor(M, x)
    S = sum(ginv .* Ric)
    return S
end

"""
    gaussian_curvature(M::MetricManifold, x)

Compute the Gaussian curvature of the manifold `M` at the point `x`.
"""
gaussian_curvature(M::MetricManifold, x) = ricci_curvature(M, x) / 2

"""
    einstein_tensor(M::MetricManifold, x)

Compute the Einstein tensor of the manifold `M` at the point `x`.
"""
function einstein_tensor(M::MetricManifold, x)
    Ric = ricci_tensor(M, x)
    g = local_metric(M, x)
    ginv = inverse_local_metric(M, x)
    S = sum(ginv .* Ric)
    G = Ric - g .* S / 2
    return G
end

@doc doc"""
    solve_exp_ode(M::MetricManifold, x, v, tspan; solver=AutoVern9(Rodas5()), kwargs...)

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
"""
function solve_exp_ode(M, x, v, tspan; kwargs...)
    error("solve_exp_ode not implemented on $(typeof(M)) for point $(typeof(x)), vector $(typeof(v)), and timespan $(typeof(tspan)). For a suitable default, enter `using OrdinaryDiffEq`.")
end
exp(M::MMT, x, v, T::AbstractVector{T} where T) where {MMT <: MetricManifold} = exp(M, is_default_metric(M), x, v, T)
exp(M::MMT, ::Val{true}, x, v, T::AbstractVector{T} where T) where {MMT<:MetricManifold} = exp(base_manifold(M), x, v, T)
function exp(M::MMT, ::Val{false}, x, v, T::AbstractVector{T} where T) where {MMT<:MetricManifold}
    sol = solve_exp_ode(M, x, v, extrema(T); dense=false, saveat=T)
    n = length(x)
    return map(i -> sol.u[i][n+1:end], 1:length(T))
end

exp!(M::MMT, y, x, v) where {MMT <: MetricManifold}= exp!(M, is_default_metric(M), y, x, v)
exp!(M::MMT, ::Val{true}, y, x, v) where {MMT<:MetricManifold} = exp!(base_manifold(M),y,x,v)
"""
    exp(MM::MetricManifold, x, v, args...)

Numerically integrate the exponential map assuming the Levi-Civita connection.
See [`solve_exp_ode`](@ref) if the [`Metric`](@ref) `G` of the [`MetricManifold`](@ref){M,G}`
is not marked as the default one.

Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.
"""
function exp!(M::MMT, ::Val{false}, y, x, v) where {MMT<:MetricManifold}
    tspan = (0.0, 1.0)
    sol = solve_exp_ode(M, x, v, tspan; dense=false, saveat=[1.0])
    n = length(x)
    y .= sol.u[1][n+1:end]
    return y
end

inner(M::MMT, x, v, w) where {MMT <: MetricManifold}= inner(M, is_default_metric(M), x, v, w)
inner(M::MMT, ::Val{false}, x, v, w) where {MMT <: MetricManifold}= dot(v, local_metric(M, x) * w)
inner(M::MMT, ::Val{true}, x, v, w) where {MMT <: MetricManifold} = inner(base_manifold(M), x,v,w)

log!(M::MMT, w, x, y) where {MMT <: MetricManifold} = log!(M, is_default_metric(M), w, x, y)
log!(M::MMT, ::Val{true}, w, x, y) where {MMT <: MetricManifold} = log!(base_manifold(M), w, x, y)
log!(M::MMT, ::Val{false}, w, x, y) where {MMT <: MetricManifold} = error("Logarithmic map not implemented on $(typeof(M)) for points $(typeof(x)) and $(typeof(y)).")

tangent_orthonormal_basis(M::MMT, x, v) where {MMT <: MetricManifold} = tangent_orthonormal_basis(M, is_default_metric(M), x, v)
tangent_orthonormal_basis(M::MMT, ::Val{true}, x, v) where {MMT <: MetricManifold} = tangent_orthonormal_basis(base_manifold(M), x, v)
tangent_orthonormal_basis(M::MMT, ::Val{false}, x, v) where {MMT <: MetricManifold} = error("tangent_orthogonal_basis not implemented on $(typeof(M)) for point $(typeof(x)) and tangent vector $(typeof(v)).")

project_point!(M::MMT, y, x) where {MMT <: MetricManifold} = project_point!(M, is_default_metric(M), y, x)
project_point!(M::MMT, ::Val{true}, y, x) where {MMT <: MetricManifold} = project_point!(base_manifold(M), y, x)
project_point!(M::MMT, ::Val{false}, y, x) where {MMT <: MetricManifold} = error("project_point! not implemented on $(typeof(M)) for point $(typeof(x))")

project_tangent!(M::MMT, w, x, v) where {MMT <: MetricManifold} = project_tangent!(M, is_default_metric(M), w, x, v)
project_tangent!(M::MMT, ::Val{true}, w, x, v) where {MMT <: MetricManifold} = project_tangent!(base_manifold(M), w, x, v)
project_tangent!(M::MMT, ::Val{false}, w, x, v) where {MMT <: MetricManifold} = error("project_tangent! not implemented for a $(typeof(M)) and tangent $(typeof(v)) at point $(typeof(x)).")

vector_transport_to!(M::MMT, vto, x, v, y, m::AbstractVectorTransportMethod) where {MMT <: MetricManifold} = vector_transport_to!(M, is_default_metric(M), vto, x, v, y, m)
vector_transport_to!(M::MMT, ::Val{true}, vto, x, v, y, m)  where {MMT <: MetricManifold}= vector_transport_to!(base_manifold(M), vto, x, v, y, m)
vector_transport_to!(M::MMT, ::Val{false}, vto, x, v, y, m)  where {MMT <: MetricManifold}= error("vector transport from a point of type $(typeof(x)) to a type $(typeof(y)) on a $(typeof(M)) for a vector of type $(v) and the $(typeof(m)) not yet implemented.")

projected_distribution(M::MMT, d, x) where {MMT <: MetricManifold} = projected_distribution(M, is_default_metric(M), d,x)
projected_distribution(M::MMT, ::Val{true}, d, x) where {MMT <: MetricManifold} = projected_distribution(base_manifold(M), d,x)
projected_distribution(M::MMT, ::Val{false}, d, x) where {MMT <: MetricManifold} =error("projected_distribution not implemented for a $(typeof(M)) and with $(typeof(d)) at point $(typeof(x)).")

projected_distribution(M::MMT, d) where {MMT <: MetricManifold} = projected_distribution(M, is_default_metric(M), d)
projected_distribution(M::MMT, ::Val{true}, d) where {MMT <: MetricManifold} = projected_distribution(base_manifold(M), d)
projected_distribution(M::MMT, ::Val{false}, d) where {MMT <: MetricManifold} =error("projected_distribution not implemented for a $(typeof(M)) with $(typeof(d)).")

normal_tvector_distribution(M::MMT, x, σ) where {MMT <: MetricManifold} = normal_tvector_distribution(M, is_default_metric(M), x, σ)
normal_tvector_distribution(M::MMT, ::Val{true}, x, σ) where {MMT <: MetricManifold} = normal_tvector_distribution(base_manifold(M), x, σ)
normal_tvector_distribution(M::MMT, ::Val{false}, x, σ) where {MMT <: MetricManifold} =error("normal_tvector_distribution not implemented for a $(typeof(M)) at point $(typeof(x)) with standard deviation $(typeof(σ)).")


function inner(B::VectorBundleFibers{<:CotangentSpaceType, MMT}, x, v, w) where {MMT<:MetricManifold}
    ginv = inverse_local_metric(B.M, x)
    return dot(v, ginv * w)
end

function flat!(M::MMT, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType}) where {MMT<:MetricManifold}
    g = local_metric(M, x)
    copyto!(v.data, g*w.data)
    return v
end

function sharp!(M::MMT, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType}) where {MMT<:MetricManifold}
    ginv = inverse_local_metric(M, x)
    copyto!(v.data, ginv*w.data)
    return v
end

# These are independent of the metric and hence can always fall back to M
injectivity_radius(M::MMT, args...) where {MMT <: MetricManifold} = injectivity_radius(base_manifold(M), args...)
zero_tangent_vector!(M::MMT, v, x) where {MMT <: MetricManifold} = zero_tangent_vector!(base_manifold(M), v, x)
check_manifold_point(M::MMT, x; kwargs...) where {MMT <: MetricManifold} = check_manifold_point(base_manifold(M), x; kwargs...)
check_tangent_vector(M::MMT, x, v; kwargs...) where {MMT <: MetricManifold} = check_tangent_vector(base_manifold(M), x, v; kwargs...)