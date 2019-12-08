module Manifolds

import Base: isapprox,
    exp,
    log,
    angle,
    eltype,
    similar,
    getindex,
    setindex!,
    size,
    length,
    copy,
    copyto!,
    convert,
    dataids,
    axes,
    promote_rule,
    +,
    -,
    *
import Distributions: mean,
    median
import LinearAlgebra: dot,
    norm,
    det,
    cross,
    I,
    UniformScaling,
    Diagonal
using Random: randperm


using ManifoldsBase
using ManifoldsBase: AbstractRetractionMethod,
    AbstractInverseRetractionMethod,
    AbstractVectorTransportMethod
import ManifoldsBase: base_manifold,
    check_manifold_point,
    check_tangent_vector,
    distance,
    exp,
    exp!,
    geodesic,
    shortest_geodesic,
    injectivity_radius,
    inner,
    inverse_retract,
    inverse_retract!,
    isapprox,
    is_manifold_point,
    is_tangent_vector,
    is_decorator_manifold,
    log,
    log!,
    manifold_dimension,
    check_manifold_point,
    norm,
    project_point,
    project_point!,
    project_tangent,
    project_tangent!,
    representation_size,
    retract,
    retract!,
    check_tangent_vector,
    vector_transport_along,
    vector_transport_along!,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
    vector_transport_to!,
    zero_tangent_vector,
    zero_tangent_vector!

using Requires
using StaticArrays
using HybridArrays
using Markdown: @doc_str
import Distributions: _rand!, support
import Random: rand
using LinearAlgebra
using Random: AbstractRNG
using UnsafeArrays
using Einsum: @einsum

hat!(M::Manifold, v, x, vⁱ) = error("hat! operator not defined for manifold $(typeof(M)), vector $(typeof(vⁱ)), and matrix $(typeof(v))")

@doc doc"""
    hat(M::Manifold, x, vⁱ)

Given a basis $e_i$ on the tangent space at a point $x$ and tangent
component vector $v^i$, compute the equivalent vector representation
$v=v^i e_i$, where Einstein summation notation is used:

$\wedge: v^i \mapsto v^i e_i$

For matrix manifolds, this converts a vector representation of the tangent
vector to a matrix representation. The [`vee`](@ref) map is the `hat` map's
inverse.
"""
function hat(M::Manifold, x, vⁱ)
    repr_size = representation_size(TangentBundleFibers(M))
    v = MArray{Tuple{repr_size...},eltype(vⁱ)}(undef)
    hat!(M, v, x, vⁱ)
    return v
end

@doc doc"""
    mean(M,x; weights=1/n*ones(n), stop_tol=10^-8, stop_iter=100 )

computes the Riemannian center of mass also known as Karcher mean of the
[`Vector`](@ref) 'x'  of points on the [`Manifold`](@ref) `M`. This function
uses the gradient descent scheme. Optionally one can provide
weights $w_i$ for the weighted Riemannian center of mass.
The general formula to compute the minimizer reads
````math
\argmin{y\in\mathcal M} \frac{1}{2}\sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}^2(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemian [`distance`](@ref).

Finally `stop_tol` denotes the minimal change between to iterates then to stop
and `stop_iter` the maximal number of iterations. For more stopping criteria
check the [`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

The algorithm is further described in 
> Afsari, B; Tron, R.; Vidal, R.: On the Convergence of Gradient
> Descent for Finding the Riemannian Center of Mass,
> SIAM Journal on Control and Optimization, Volume 51, Issue 3,
> pp. 2230–2260, 
> doi: [10.1137/12086282X](https://doi.org/10.1137/12086282X),
> arxiv: [1201.0925](https://arxiv.org/abs/1201.0925">1201.0925)
"""
function mean(M::Manifold, x::Vector{T};
    weights=1/length(x)*ones(length(x)),
    stop_atol=10^-7,
    stop_iter=100
    ) where {T}
    iter = 0
    y=x[1]
    yOld = y
    while ( ( iter==0 || distance(M,y,yOld) > stop_atol ) && (iter < stop_iter) )
        iter += 1
        yOld = y
        y = exp(M,yOld,  1/2*sum( weights.*log.(Ref(M), Ref(yOld),x) )  )
    end
    return y
end

@doc doc"""
    median(M,x; weights=1/n*ones(n), stop_tol=10^-8, stop_iter=100, use_rand = true )

computes the Riemannian median of the [`Vector`](@ref) 'x'  of points on the
[`Manifold`](@ref) `M`. This function is nonsmooth (i.e nondifferentiable) and
uses a cyclic procimal point scheme. Optionally one can provide
weights $w_i$ for the weighted Riemannian median. The general formula to compute
the minimizer reads
````math
\argmin{y\in\mathcal M}\sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemian [`distance`](@ref).

Finally `stop_tol` denotes the minimal change between to iterates then to stop
and `stop_iter` the maximal number of iterations, and the cycle order of the
cyclic proximal point method can be set to `random`, which sometimes performes
faster.  For more stopping criterai and details check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

The algorithm is further described in Algorithm 4.3 and 4.4 in 
> Bačák, M: Computing Medians and Means in Hadamard Spaces.
> SIAM Journal on Optimization, Volume 24, Number 3, pp. 1542–1566,
> doi: [10.1137/140953393](https://doi.org/10.1137/140953393),
> arxiv: [1210.2145](https://arxiv.org/abs/1210.2145>
"""
function median(M::Manifold, x::Vector{T};
    weights=1/length(x)*ones(length(x)),
    stop_atol=10^-10,
    stop_iter=10000,
    use_random = false
    ) where {T}
    n = length(x)
    y=x[1]
    yOld = y
    order = 1:n
    print(order)
    iter = 0
    while ( ( iter==0 || distance(M,y,yOld) > stop_atol ) && (iter < stop_iter) )
        iter += 1
        λ = 1/(iter+1)
        yOld = y
        order = use_random ? randperm(n) : order
        for i=1:n
            t = min( λ * weights[order[i]] / distance(M,y,x[order[i]]) , 1 )
            y = exp( M, y, t*log(M, y, x[order[i]]) )
        end
    end
    return y
end
vee!(M::Manifold, vⁱ, x, v) = error("vee! operator not defined for manifold $(typeof(M)), matrix $(typeof(v)), and vector $(typeof(vⁱ))")

@doc doc"""
    vee(M::Manifold, x, v)

Given a basis $e_i$ on the tangent space at a point $x$ and tangent
vector $v$, compute the vector components $v^i$, such that $v = v^i e_i$, where
Einstein summation notation is used:

$\vee: v^i e_i \mapsto v^i$

For matrix manifolds, this converts a  matrix representation of the tangent
vector to a vector representation. The [`hat`](@ref) map is the `vee` map's
inverse.
"""
function vee(M::Manifold, x, v)
    vⁱ = MVector{manifold_dimension(M),eltype(v)}(undef)
    vee!(M, vⁱ, x, v)
    return vⁱ
end

include("utils.jl")
include("SizedAbstractArray.jl")

include("ProductRepresentations.jl")
include("VectorBundle.jl")
include("Metric.jl")

include("DistributionsBase.jl")
include("ProjectedDistribution.jl")

include("ProductManifold.jl")
include("PowerManifold.jl")

include("Euclidean.jl")
include("CholeskySpace.jl")
include("Rotations.jl")
include("Sphere.jl")
include("SymmetricPositiveDefinite.jl")

function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
        using .ForwardDiff
        include("forward_diff.jl")
    end

    @require OrdinaryDiffEq="1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        using .OrdinaryDiffEq: ODEProblem,
            AutoVern9,
            Rodas5,
            solve
        include("ode.jl")
    end
end

export ArrayManifold,
    ArrayMPoint,
    ArrayTVector

export Manifold,
    MPoint,
    TVector,
    ProductManifold,
    PowerManifold,
    ProductRepr,
    VectorSpaceType,
    TangentSpace,
    CotangentSpace,
    VectorSpaceAtPoint,
    TangentSpaceAtPoint,
    CotangentSpaceAtPoint,
    VectorBundle,
    VectorBundleFibers,
    FVector,
    TangentBundle,
    CotangentBundle,
    TangentBundleFibers,
    CotangentBundleFibers,
    AbstractVectorTransportMethod,
    ParallelTransport,
    ProjectTangent,
    ProjectedPointDistribution
export
    Euclidean,
    CholeskySpace,
    Sphere,
    SymmetricPositiveDefinite

export base_manifold,
    bundle_projection,
    distance,
    exp,
    exp!,
    flat,
    flat!,
    hat!,
    hat,
    sharp,
    sharp!,
    vee,
    vee!,
    geodesic,
    shortest_geodesic,
    injectivity_radius,
    inverse_retract,
    inverse_retract!,
    is_default_metric,
    is_manifold_point,
    is_tangent_vector,
    isapprox,
    inner,
    log,
    log!,
    manifold_dimension,
    mean,
    median,
    norm,
    normal_tvector_distribution,
    project_point,
    project_point!,
    project_tangent,
    project_tangent!,
    projected_distribution,
    representation_size,
    retract,
    retract!,
    submanifold,
    submanifold_component,
    tangent_orthonormal_basis,
    vector_space_dimension,
    vector_transport_along,
    vector_transport_along!,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
    vector_transport_to!,
    zero_vector,
    zero_vector!,
    zero_tangent_vector,
    zero_tangent_vector!
export Metric,
    RiemannianMetric,
    LorentzMetric,
    EuclideanMetric,
    MetricManifold,
    LinearAffineMetric,
    LogEuclideanMetric,
    LogCholeskyMetric,
    metric,
    local_metric,
    inverse_local_metric,
    local_metric_jacobian,
    det_local_metric,
    log_local_metric_density,
    christoffel_symbols_first,
    christoffel_symbols_second,
    christoffel_symbols_second_jacobian,
    riemann_tensor,
    ricci_tensor,
    einstein_tensor,
    ricci_curvature,
    gaussian_curvature

end # module
