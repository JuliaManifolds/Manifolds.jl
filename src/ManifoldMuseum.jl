module ManifoldMuseum

import Base: isapprox,
    exp,
    log,
    angle,
    eltype,
    similar,
    convert,
    +,
    -,
    *
import LinearAlgebra: dot,
    norm,
    det,
    I,
    UniformScaling,
    Diagonal
import Markdown: @doc_str
import Distributions: _rand!
import Random: rand
using Random: AbstractRNG
using SimpleTraits
using ForwardDiff
import TensorOperations: @tensor
import DifferentialEquations: ODEProblem,
    AutoVern9,
    Rodas5,
    solve

"""
    Manifold

A manifold type. The `Manifold` is used to dispatch to different exponential
and logarithmic maps as well as other function on manifold.
"""
abstract type Manifold end

"""
    MPoint

Type for a point on a manifold. While a [`Manifold`](@ref) not necessarily
requires this type, for example when it is implemented for `Vector`s or
`Matrix` type elements, this type can be used for more complicated
representations, semantic verification or even dispatch for different
representations of points on a manifold.
"""
abstract type MPoint end

"""
    TVector

Type for a tangent vector of a manifold. While a [`Manifold`](@ref) not
necessarily requires this type, for example when it is implemented for `Vector`s
or `Matrix` type elements, this type can be used for more complicated
representations, semantic verification or even dispatch for different
representations of tangent vectors and their types on a manifold.
"""
abstract type TVector end

"""
    CoTVector

Type for a cotangent vector of a manifold. While a [`Manifold`](@ref) not
necessarily requires this type, for example when it is implemented for `Vector`s
or `Matrix` type elements, this type can be used for more complicated
representations, semantic verification or even dispatch for different
representations of cotangent vectors and their types on a manifold.
"""
abstract type CoTVector end

"""
    IsDecoratorManifold

A `Trait` to mark a manifold as a decorator type. For any function that is only
implemented for a decorator (i.e. a Manifold with `@traitimpl
IsDecoratorManifold{M}`), a specific function should be implemented as a
`@traitfn`, that transparently passes down through decorators, i.e.

```
@traitfn myFeature(M::Mt, k...) where {Mt; IsDecoratorManifold{Mt}} = myFeature(M.manifold, k...)
```
or the shorter version
```
@traitfn myFeature(M::::IsDecoratorManifold, k...) = myFeature(M.manifold, k...)
```
such that decorators act just as pass throughs for other decorator functions and
```
myFeature(M::MyManifold, k...) = #... my explicit implementation
```
then implements the feature itself.
"""
@traitdef IsDecoratorManifold{M}

"""
    base_manifold(M::Manifold)

Strip all decorators on `M`, returning the underlying topological manifold.
"""
function manifold end

@traitfn function base_manifold(M::MT) where {MT<:Manifold;IsDecoratorManifold{MT}}
    return base_manifold(M.manifold)
end

@traitfn base_manifold(M::MT) where {MT<:Manifold;!IsDecoratorManifold{MT}} = M

@doc doc"""
    manifold_dimension(M::Manifold)

The dimension $n$ of real space $\mathbb R^n$ to which the neighborhood
of each point of the manifold is homeomorphic.
"""
function manifold_dimension end

@traitfn function manifold_dimension(M::MT) where {MT<:Manifold;!IsDecoratorManifold{MT}}
    error("manifold_dimension not implemented for a $(typeof(M)).")
end

@traitfn function manifold_dimension(M::MT) where {MT<:Manifold;IsDecoratorManifold{MT}}
    manifold_dimension(base_manifold(M))
end

"""
    isapprox(M::Manifold, x, y; kwargs...)

Check if points `x` and `y` from manifold `M` are approximately equal.

Keyword arguments can be used to specify tolerances.
"""
isapprox(M::Manifold, x, y; kwargs...) = isapprox(x, y; kwargs...)

"""
    isapprox(M::Manifold, x, v, w; kwargs...)

Check if vectors `v` and `w` tangent at `x` from manifold `M` are
approximately equal.

Keyword arguments can be used to specify tolerances.
"""
isapprox(M::Manifold, x, v, w; kwargs...) = isapprox(v, w; kwargs...)

"""
    retract!(M::Manifold, y, x, v, t=1)

Retraction (cheaper, approximate version of exponential map) of tangent
vector `t*v` at point `x` from manifold `M`.
Result is saved to `y`.
"""
retract!(M::Manifold, y, x, v) = exp!(M, y, x, v)

retract!(M::Manifold, y, x, v, t) = retract!(M, y, x, t*v)

"""
    retract(M::Manifold, x, v, t=1)

Retraction (cheaper, approximate version of exponential map) of tangent
vector `t*v` at point `x` from manifold `M`.
"""
function retract(M::Manifold, x, v)
    xr = similar_result(M, retract, x, v)
    retract!(M, xr, x, v)
    return xr
end

retract(M::Manifold, x, v, t) = retract(M, x, t*v)

project_tangent!(M::Manifold, w, x, v) = error("project onto tangent space not implemented for a $(typeof(M)) and point $(typeof(x)) with input $(typof(v)).")

function project_tangent(M::Manifold, x, v)
    vt = similar_result(M, project_tangent, v, x)
    project_tangent!(M, vt, x, v)
    return vt
end

distance(M::Manifold, x, y) = norm(M, x, log(M, x, y))

"""
    inner(M::Manifold, x, v, w)

Inner product of tangent vectors `v` and `w` at point `x` from manifold `M`.
"""
inner(M::Manifold, x, v, w) = error("inner: Inner product not implemented on a $(typeof(M)) for input point $(typeof(x)) and tangent vectors $(typeof(v)) and $(typeof(w)).")

"""
    norm(M::Manifold, x, v)

Norm of tangent vector `v` at point `x` from manifold `M`.
"""
norm(M::Manifold, x, v) = sqrt(inner(M, x, v, v))

"""
    angle(M::Manifold, x, v, w)

Angle between tangent vectors `v` and `w` at point `x` from manifold `M`.
"""
angle(M::Manifold, x, v, w) = acos(inner(M, x, v, w) / norm(M, x, v) / norm(M, x, w))

"""
    exp!(M::Manifold, y, x, v, t=1)

Exponential map of tangent vector `t*v` at point `x` from manifold `M`.
Result is saved to `y`.
"""
exp!(M::Manifold, y, x, v, t) = exp!(M::Manifold, y, x, t*v)

exp!(M::Manifold, y, x, v) = error("Exponential map not implemented on a $(typeof(M)) for input point $(x) and tangent vector $(v).")

"""
    exp(M::Manifold, x, v, t=1)

Exponential map of tangent vector `t*v` at point `x` from manifold `M`.
"""
function exp(M::Manifold, x, v)
    x2 = similar_result(M, x, v)
    exp!(M, x2, x, v)
    return x2
end

exp(M::Manifold, x, v, t) = exp(M, x, t*v)

log!(M::Manifold, v, x, y) = error("Logarithmic map not implemented on $(typeof(M)) for points $(typeof(x)) and $(typeof(y))")

function log(M::Manifold, x, y)
    v = similar_result(M, log, x, y)
    log!(M, v, x, y)
    return v
end

vector_transport!(M::Manifold, vto, x, v, y) = project_tangent!(M, vto, x, v)

function vector_transport(M::Manifold, x, v, y)
    vto = similar_result(M, vector_transport, v, x, y)
    vector_transport!(M, vto, x, y, v)
    return vto
end

"""
    injectivity_radius(M::Manifold, x)

Distance such that `log(M, x, y)` is defined for all points within this radius.
"""
injectivity_radius(M::Manifold, x) = Inf

"""
    injectivity_radius(M::Manifold, x)

Infimum of the injectivity radii of all manifold points.
"""
injectivity_radius(M::Manifold) = Inf

zero_tangent_vector(M::Manifold, x) = log(M, x, x)
zero_tangent_vector!(M::Manifold, v, x) = log!(M, v, x, x)

geodesic(M::Manifold, x, y, t) = exp(M, x, log(M, x, y), t)

"""
    similar_result_type(M::Manifold, f, args::NTuple{N,Any}) where N

Returns type of element of the array that will represent the result of
function `f` for manifold `M` on given arguments (passed at a tuple)
"""
function similar_result_type(M::Manifold, f, args::NTuple{N,Any}) where N
    T = typeof(reduce(+, one(eltype(eti)) for eti ∈ args))
    return T
end

"""
    similar_result(M::Manifold, f, x...)

Allocates an array for the result of function `f` on manifold `M`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
function similar_result(M::Manifold, f, x...)
    T = similar_result_type(M, f, x)
    return similar(x[1], T)
end

"""
    is_manifold_point(M,x)

check, whether `x` is a valid point on the [`Manifold`](@ref) `M`. If it is not,
an error is thrown.
The default is to return `true`, i.e. if no checks are implmented,
the assumption is to be optimistic.
"""
is_manifold_point(M::Manifold, x; kwargs...) = true
is_manifold_point(M::Manifold, x::MPoint) = error("A validation for a $(typeof(x)) on $(typeof(M)) not implemented.")

"""
    is_tangent_vector(M,x,v)

check, whether `v` is a valid tangnt vector in the tangent plane of `x` on the
[`Manifold`](@ref) `M`. An implementation should first check
[`is_manifold_point`](@ref)`(M,x)` and then validate `v`. If it is not a tangent
vector an error should be thrown.
The default is to return `true`, i.e. if no checks are implmented,
the assumption is to be optimistic.
"""
is_tangent_vector(M::Manifold, x, v; kwargs...) = true
is_tangent_vector(M::Manifold, x::MPoint, v::TVector) = error("A validation for a $(typeof(v)) in the tangent space of a $(typeof(x)) on $(typeof(M)) not implemented.")

include("ArrayManifold.jl")

include("DistributionsBase.jl")
include("Metric.jl")
include("Euclidean.jl")
include("Sphere.jl")
include("ProjectedDistribution.jl")

export Manifold,
    IsDecoratorManifold,
    Euclidean
export manifold_dimension,
    base_manifold,
    distance,
    inner,
    exp,
    exp!,
    geodesic,
    isapprox,
    log,
    log!,
    norm,
    injectivity_radius,
    zero_tangent_vector,
    zero_tangent_vector!
export Metric,
    RiemannianMetric,
    LorentzMetric,
    EuclideanMetric,
    MetricManifold,
    HasMetric,
    metric,
    local_metric,
    inverse_local_metric,
    det_local_metric,
    christoffel_symbols_first,
    christoffel_symbols_second,
    riemann_tensor,
    ricci_tensor,
    einstein_tensor,
    ricci_curvature,
    gaussian_curvature

end # module
