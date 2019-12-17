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
import Statistics: mean,
    mean!,
    median,
    median!,
    var,
    std
import StatsBase: mean_and_std,
    mean_and_var
import LinearAlgebra: dot,
    norm,
    det,
    cross,
    I,
    UniformScaling,
    Diagonal

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
    zero_tangent_vector!,
    similar_result

using Requires
using StaticArrays
using HybridArrays
using Markdown: @doc_str
import Distributions: _rand!, support
import Random: rand
using LinearAlgebra
using Random: AbstractRNG
using FiniteDifferences
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

include("Statistics.jl")

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
    mean!,
    mean_and_var,
    mean_and_std,
    median,
    median!,
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
    std,
    submanifold,
    submanifold_component,
    tangent_orthonormal_basis,
    var,
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
