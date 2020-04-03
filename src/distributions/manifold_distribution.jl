"""
    normal_tvector_distribution(M::Euclidean, p, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `p`.
"""
function normal_tvector_distribution(M::Euclidean{Tuple{N}}, p, σ) where {N}
    d = Distributions.MvNormal(zero(p), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), p, d, project_vector!, p)
end

"""
    projected_distribution(M::Euclidean, d, [p])

Wrap the standard distribution `d` into a manifold-valued distribution. Generated
points will be of similar type to `p`. By default, the type is not changed.
"""
function projected_distribution(M::Euclidean, d, p)
    return ProjectedPointDistribution(M, d, project!, p)
end
function projected_distribution(M::Euclidean, d)
    return ProjectedPointDistribution(M, d, project!, rand(d))
end

function ManifoldsBase.decorator_transparent_dispatch(::typeof(normal_tvector_distribution), M::MetricManifold, arge...)
    return Val(:intransparent)
end
function ManifoldsBase.decorator_transparent_dispatch(::typeof(projected_distribution), M::MetricManifold, arge...)
    return Val(:intransparent)
end

"""
    PowerPointDistribution(M::AbstractPowerManifold, distribution)

Power distribution on manifold `M`, based on `distribution`.
"""
struct PowerPointDistribution{TM<:AbstractPowerManifold,TD<:MPointDistribution,TX} <:
       MPointDistribution{TM}
    manifold::TM
    distribution::TD
    point::TX
end

"""
    PowerFVectorDistribution([type::VectorBundleFibers], [x], distr)

Generates a random vector at a `point` from vector space (a fiber of a tangent
bundle) of type `type` using the power distribution of `distr`.

Vector space type and `point` can be automatically inferred from distribution `distr`.
"""
struct PowerFVectorDistribution{
    TSpace<:VectorBundleFibers{<:VectorSpaceType,<:AbstractPowerManifold},
    TD<:FVectorDistribution,
    TX,
} <: FVectorDistribution{TSpace,TX}
    type::TSpace
    point::TX
    distribution::TD
end

function Random.rand(rng::AbstractRNG, d::PowerFVectorDistribution)
    fv = zero_vector(d.type, d.point)
    Distributions._rand!(rng, d, fv)
    return fv
end
function Random.rand(rng::AbstractRNG, d::PowerPointDistribution)
    x = allocate_result(d.manifold, rand, d.point)
    Distributions._rand!(rng, d, x)
    return x
end

function Distributions._rand!(rng::AbstractRNG, d::PowerFVectorDistribution, v::AbstractArray)
    PM = d.type.manifold
    rep_size = representation_size(PM.manifold)
    for i in get_iterator(d.type.manifold)
        copyto!(d.distribution.point, _read(PM, rep_size, d.point, i))
        Distributions._rand!(rng, d.distribution, _read(PM, rep_size, v, i))
    end
    return v
end
function Distributions._rand!(rng::AbstractRNG, d::PowerPointDistribution, x::AbstractArray)
    M = d.manifold
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        Distributions._rand!(rng, d.distribution, _write(M, rep_size, x, i))
    end
    return x
end

Distributions.support(tvd::PowerFVectorDistribution) = FVectorSupport(tvd.type, tvd.point)
Distributions.support(d::PowerPointDistribution) = MPointSupport(d.manifold)

"""
    ProductFVectorDistribution([type::VectorBundleFibers], [x], distrs...)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the product distribution of given distributions.

Vector space type and `x` can be automatically inferred from distributions `distrs`.
"""
struct ProductFVectorDistribution{
    TSpace<:VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    TD<:(NTuple{N,Distribution} where {N}),
    TX,
} <: FVectorDistribution{TSpace,TX}
    type::TSpace
    x::TX
    distributions::TD
end

"""
    ProductPointDistribution(M::ProductManifold, distributions)

Product distribution on manifold `M`, combined from `distributions`.
"""
struct ProductPointDistribution{
    TM<:ProductManifold,
    TD<:(NTuple{N,Distribution} where {N}),
} <: MPointDistribution{TM}
    manifold::TM
    distributions::TD
end

function ProductFVectorDistribution(
    type::VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    p::Union{AbstractArray,MPoint,ProductRepr},
    distributions::FVectorDistribution...,
)
    return ProductFVectorDistribution{typeof(type),typeof(distributions),typeof(p)}(
        type,
        p,
        distributions,
    )
end
function ProductFVectorDistribution(
    type::VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    distributions::FVectorDistribution...,
)
    p = ProductRepr(map(d -> support(d).point, distributions))
    return ProductFVectorDistribution(type, p, distributions...)
end
function ProductFVectorDistribution(distributions::FVectorDistribution...)
    M = ProductManifold(map(d -> support(d).space.manifold, distributions)...)
    fiber = support(distributions[1]).space.fiber
    if !all(d -> support(d).space.fiber == fiber, distributions)
        error("Not all distributions have support in vector spaces of the same type, which is currently not supported")
    end
    # Probably worth considering sum spaces in the future?
    x = ProductRepr(map(d -> support(d).point, distributions)...)
    return ProductFVectorDistribution(VectorBundleFibers(fiber, M), x, distributions...)
end

function ProductPointDistribution(M::ProductManifold, distributions::MPointDistribution...)
    return ProductPointDistribution{typeof(M),typeof(distributions)}(M, distributions)
end
function ProductPointDistribution(distributions::MPointDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    return ProductPointDistribution(M, distributions...)
end

function Random.rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ProductRepr(map(d -> rand(rng, d), d.distributions)...)
end
function Random.rand(rng::AbstractRNG, d::ProductFVectorDistribution)
    return ProductRepr(map(d -> rand(rng, d), d.distributions)...)
end

function Distributions._rand!(rng::AbstractRNG, d::ProductPointDistribution, x::AbstractArray{<:Number})
    return copyto!(x, rand(rng, d))
end
function Distributions._rand!(rng::AbstractRNG, d::ProductPointDistribution, p::ProductRepr)
    map(
        t -> Distributions._rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.manifold, p),
    )
    return p
end
function Distributions._rand!(rng::AbstractRNG, d::ProductFVectorDistribution, v::AbstractArray{<:Number})
    return copyto!(v, rand(rng, d))
end
function Distributions._rand!(rng::AbstractRNG, d::ProductFVectorDistribution, X::ProductRepr)
    map(
        t -> Distributions._rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.space.manifold, X),
    )
    return X
end

Distributions.support(d::ProductPointDistribution) = MPointSupport(d.manifold)
function Distributions.support(tvd::ProductFVectorDistribution)
    return FVectorSupport(
        tvd.type,
        ProductRepr(map(d -> support(d).point, tvd.distributions)...),
    )
end

"""
    NormalRotationDistribution(M::Rotations, d::Distribution, x::TResult)

Distribution that returns a random point on the manifold [`Rotations`](@ref)
`M`. Random point is generated using base distribution `d` and the type
of the result is adjusted to `TResult`.

See [`normal_rotation_distribution`](@ref) for details.
"""
struct NormalRotationDistribution{TResult,TM<:Rotations,TD<:Distribution} <:
       MPointDistribution{TM}
    manifold::TM
    distr::TD
end

function NormalRotationDistribution(
    M::Rotations,
    d::Distribution,
    x::TResult,
) where {TResult}
    return NormalRotationDistribution{TResult,typeof(M),typeof(d)}(M, d)
end

@doc raw"""
    normal_rotation_distribution(M::Rotations, p, σ::Real)

Return a random point on the manifold [`Rotations`](@ref) `M`
by generating a (Gaussian) random orthogonal matrix with determinant $+1$. Let

$QR = A$

be the QR decomposition of a random matrix $A$, then the formula reads

$p = QD$

where $D$ is a diagonal matrix with the signs of the diagonal entries of $R$,
i.e.

$D_{ij}=\begin{cases} \operatorname{sgn}(R_{ij}) & \text{if} \; i=j \\ 0 & \, \text{otherwise} \end{cases}.$

It can happen that the matrix gets -1 as a determinant. In this case, the first
and second columns are swapped.

The argument `p` is used to determine the type of returned points.
"""
function normal_rotation_distribution(M::Rotations{N}, p, σ::Real) where {N}
    d = Distributions.MvNormal(zeros(N * N), σ)
    return NormalRotationDistribution(M, d, p)
end

"""
    normal_tvector_distribution(M::Rotations, p, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `p`.
"""
function normal_tvector_distribution(M::Rotations, p, σ)
    d = Distributions.MvNormal(reshape(zero(p), :), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), p, d, project_vector!, p)
end

function Random.rand(
    rng::AbstractRNG,
    d::NormalRotationDistribution{TResult,Rotations{N}},
) where {TResult,N}
    if N == 1
        return convert(TResult, ones(1, 1))
    else
        A = reshape(rand(rng, d.distr), (N, N))
        return convert(TResult, _fix_random_rotation(A))
    end
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::NormalRotationDistribution{TResult,Rotations{N}},
    x::AbstractArray{<:Real},
) where {TResult,N}
    return copyto!(x, rand(rng, d))
end

function _fix_random_rotation(A::AbstractMatrix)
    s = diag(sign.(qr(A).R))
    D = Diagonal(s)
    C = qr(A).Q * D
    if det(C) < 0
        C[:, [1, 2]] = C[:, [2, 1]]
    end
    return C
end

"""
    normal_tvector_distribution(S::Sphere, p, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `p`.
"""
function normal_tvector_distribution(S::Sphere, p, σ)
    d = Distributions.MvNormal(zero(p), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(S), p, d, project_vector!, p)
end

"""
    uniform_distribution(M::Sphere, p)

Uniform distribution on given [`Sphere`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::Sphere, p)
    d = Distributions.MvNormal(zero(p), 1.0)
    return ProjectedPointDistribution(M, d, project!, p)
end
