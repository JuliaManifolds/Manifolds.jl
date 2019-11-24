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

Equip a manifold with a metric. Such a manifold is generally called pseudo-
or semi-Riemannian. Each `MetricManifold` must implement
[`local_metric`](@ref).

# Constructor

    MetricManifold(manifold, metric)
"""
struct MetricManifold{M<:Manifold,G<:Metric} <: Manifold
    manifold::M
    metric::G
end

convert(::Type{MT},M::MetricManifold{MT,GT}) where {MT,GT} = M.manifold

@traitimpl IsDecoratorManifold{MetricManifold}

"""
    DefaultMetric

A Trait to indicate that a metric `G` is the default metric for a `Manifold`
`M`, i.e. as `M::Manifold` being shorthand for a
`MetricManifold{M,G}`. this is accomplished by introducing conversions.

Accordingly you have to 

```
struct MyManifold{T} <: Manifold end
struct MyMetric{S} <: Metric end
@traitimpl DefaultMetric{MyManifold,MyMetric}
```

and then only implent your functions for `MyManifold` when you actually refer
to `MetricManifold{MyManifold{T},MyMetric{S}}`.
"""
@traitdef DefaultMetric{M,G}

# Make this metric default, i.e. automatically convert
@traitfn convert(::Type{MT}, M::MMT) where {MT<:Manifold,
                                   GT<:Metric,
                                   MMT<:MetricManifold{MT,GT};
                                   DefaultMetric{MT,GT}} = M.manifold
@traitfn convert(::Type{MMT}, M::MT) where {MT<:Manifold,
                                   GT<:Metric,
                                   MMT<:MetricManifold{MT,GT};
                                   DefaultMetric{MT,GT}} = MetricManifold(M, GT())

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


@traitfn function inner(M::MMT, x, v, w) where {MT<:Manifold,
                                                GT<:Metric,
                                                MMT<:MetricManifold{MT,GT};
                                                !DefaultMetric{MT,GT}}
    return dot(v, local_metric(M, x) * w)
end

@traitfn function inner(M::MMT, x, v, w) where {MT<:Manifold,
                                                GT<:Metric,
                                                MMT<:MetricManifold{MT,GT};
                                                DefaultMetric{MT,GT}}
    return inner(M.manifold, x, v, w)
end

@traitfn function inner(B::VectorBundleFibers{<:CotangentSpaceType, MMT}, x, v, w) where {MT<:Manifold,
                                                                                          GT<:Metric,
                                                                                          MMT<:MetricManifold{MT,GT};
                                                                                          !DefaultMetric{MT,GT}}
    ginv = inverse_local_metric(B.M, x)
    return dot(v, ginv * w)
end

@traitfn function inner(B::VectorBundleFibers{<:CotangentSpaceType, MMT}, x, v, w) where {MT<:Manifold,
                                                                                          GT<:Metric,
                                                                                          MMT<:MetricManifold{MT,GT};
                                                                                          DefaultMetric{MT,GT}}
    return inner(VectorBundleFibers(B.VS, B.M.manifold), x, v, w)
end

@traitfn function norm(M::MMT, x, v) where {MT<:Manifold,
                                            GT<:Metric,
                                            MMT<:MetricManifold{MT,GT};
                                            !DefaultMetric{MT,GT}}
    return sqrt(inner(M, x, v, v))
end

@traitfn function norm(M::MMT, x, v) where {MT<:Manifold,
                                            GT<:Metric,
                                            MMT<:MetricManifold{MT,GT};
                                            DefaultMetric{MT,GT}}
    return norm(M.manifold, x, v)
end

@traitfn function distance(M::MMT, x, y) where {MT<:Manifold,
                                                GT<:Metric,
                                                MMT<:MetricManifold{MT,GT};
                                                !DefaultMetric{MT,GT}}
    return norm(M, x, log(M, x, y))
end


@traitfn function distance(M::MMT, x, y) where {MT<:Manifold,
                                                GT<:Metric,
                                                MMT<:MetricManifold{MT,GT};
                                                DefaultMetric{MT,GT}}
    return distance(M.manifold, x, y)
end

function zero_tangent_vector(M::MMT, x, v) where {MT<:Manifold,
                                            GT<:Metric,
                                            MMT<:MetricManifold{MT,GT}}
    return zero_tangent_vector(M.manifold, x, v, v)
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
    solve_exp_ode(M::MetricManifold,
                  x,
                  v,
                  tspan;
                  solver=AutoVern9(Rodas5()),
                  kwargs...)

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
    error("solve_exp_ode not implemented on $(typeof(M)) for point $(typeof(x)), vector $(typeof(y)), and timespan $(typeof(tspan)). For a suitable default, enter `using OrdinaryDiffEq`.")
end

@traitfn function exp(M::MMT,
                      x,
                      v,
                      T::AbstractVector) where {MT<:Manifold,
                                                GT<:Metric,
                                                MMT<:MetricManifold{MT,GT};
                                                !DefaultMetric{MT,GT}}
    sol = solve_exp_ode(M, x, v, extrema(T); dense=false, saveat=T)
    n = length(x)
    return map(i -> sol.u[i][n+1:end], 1:length(T))
end

"""
    exp(M::MetricManifold, x, v, args...)

If the [`DefaultMetric`](@ref) trait is defined for `M`, compute the exponential
map of the base manifold. Otherwise, numerically integrate the exponential
map assuming the Levi-Civita connection. See [`solve_exp_ode`](@ref)

Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.
"""
function exp end

# @traitfn function exp!(M:::MetricManifold{MT,GT}, y, x, v) where {MT<:Manifold,
#                                                GT<:Metric;
#                                                !DefaultMetric{MT,GT}}
#     tspan = (0.0, 1.0)
#     sol = solve_exp_ode(M, x, v, tspan; dense=false, saveat=[1.0])
#     n = length(x)
#     y .= sol.u[1][n+1:end]
#     return y
# end

@traitfn function exp!(M::MMT, y, x, v) where {MT<:Manifold,
                                               GT<:Metric,
                                               MMT<:MetricManifold{MT,GT};
                                               DefaultMetric{MT,GT}}
    return exp!(M.manifold, y, x, v)
end

@traitfn function log!(M::MMT, v, x, y) where {MT<:Manifold,
                                               GT<:Metric,
                                               MMT<:MetricManifold{MT,GT};
                                               DefaultMetric{MT,GT}}
    return log!(M.manifold, v, x, y)
end

@traitfn function retract!(M::MMT,
                           y,
                           args...) where {MT<:Manifold,
                                           GT<:Metric,
                                           MMT<:MetricManifold{MT,GT};
                                           DefaultMetric{MT,GT}}
    return retract!(M.manifold, y, arg...)
end

@traitfn function project_tangent!(M::MMT,
                                   w,
                                   x,
                                   v) where {MT<:Manifold,
                                             GT<:Metric,
                                             MMT<:MetricManifold{MT,GT};
                                             DefaultMetric{MT,GT}}
    return project_tangent!(M.manifold, w, x, v)
end

@traitfn function vector_transport_to!(M::MMT, vto, x, v, y, m::AbstractVectorTransportMethod) where {MT<:Manifold,
                                                     GT<:Metric,
                                                     MMT<:MetricManifold{MT,GT};
                                                     DefaultMetric{MT,GT}}
    return vector_transport_to!(M.manifold, vto, x, v, y, m)
end

@traitfn function injectivity_radius(M::MMT,
                                     args...) where {MT<:Manifold,
                                                     GT<:Metric,
                                                     MMT<:MetricManifold{MT,GT};
                                                     DefaultMetric{MT,GT}}
    return injectivity_radius(M.manifold, args...)
end

@traitfn function zero_tangent_vector!(M::MMT,
                                       v,
                                       x) where {MT<:Manifold,
                                                 GT<:Metric,
                                                 MMT<:MetricManifold{MT,GT};
                                                 DefaultMetric{MT,GT}}
    return zero_tangent_vector!(M.manifold, v, x)
end

@traitfn function is_manifold_point(M::MMT,
                                    x;
                                    kwargs...) where {MT<:Manifold,
                                                      GT<:Metric,
                                                      MMT<:MetricManifold{MT,GT};
                                                      DefaultMetric{MT,GT}}
    return is_manifold_point(M.manifold, x; kwargs...)
end

@traitfn function is_tangent_vector(M::MMT,
                                    x,
                                    v;
                                    kwargs...) where {MT<:Manifold,
                                                      GT<:Metric,
                                                      MMT<:MetricManifold{MT,GT};
                                                      DefaultMetric{MT,GT}}
    return is_tangent_vector(M.manifold, x, v; kwargs...)
end

@traitfn function flat!(M::MMT,
                        v::FVector{CotangentSpaceType},
                        x,
                        w::FVector{TangentSpaceType}) where {MT<:Manifold,
                                                             GT<:Metric,
                                                             MMT<:MetricManifold{MT,GT};
                                                             !DefaultMetric{MT,GT}}
    g = local_metric(M, x)
    copyto!(v, g*w)
    return v
end

@traitfn function flat!(M::MMT,
                        v::FVector{CotangentSpaceType},
                        x,
                        w::FVector{TangentSpaceType}) where {MT<:Manifold,
                                                             GT<:Metric,
                                                             MMT<:MetricManifold{MT,GT};
                                                             DefaultMetric{MT,GT}}
    return flat!(M.manifold, v, x, w)
end

@traitfn function sharp!(M::MMT,
                         v::FVector{TangentSpaceType},
                         x,
                         w::FVector{CotangentSpaceType}) where {MT<:Manifold,
                                                                GT<:Metric,
                                                                MMT<:MetricManifold{MT,GT};
                                                                !DefaultMetric{MT,GT}}
    ginv = inverse_local_metric(M, x)
    copyto!(v, ginv*w)
    return v
end

@traitfn function sharp!(M::MMT,
                         v::FVector{TangentSpaceType},
                         x,
                         w::FVector{CotangentSpaceType}) where {MT<:Manifold,
                                                                GT<:Metric,
                                                                MMT<:MetricManifold{MT,GT};
                                                                DefaultMetric{MT,GT}}
    return sharp!(M.manifold, v, x, w)
end
