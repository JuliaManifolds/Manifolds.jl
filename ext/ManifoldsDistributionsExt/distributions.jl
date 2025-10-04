"""
    FVectorvariate

Structure that subtypes `VariateForm`, indicating that a single sample
is a vector from a fiber of a vector bundle.
"""
struct FVectorvariate <: VariateForm end

"""
    FVectorSupport(space::AbstractManifold, VectorSpaceFiber)

Value support for vector bundle fiber-valued distributions (values from a fiber of a vector
bundle at a `point` from the given manifold).
For example used for tangent vector-valued distributions.
"""
struct FVectorSupport{TSpace <: VectorSpaceFiber} <: ValueSupport
    space::TSpace
end

"""
    FVectorDistribution{TSpace<:VectorSpaceFiber, T}

An abstract distribution for vector bundle fiber-valued distributions (values from a fiber
of a vector bundle at point `x` from the given manifold).
For example used for tangent vector-valued distributions.
"""
abstract type FVectorDistribution{TSpace <: VectorSpaceFiber} <:
Distribution{FVectorvariate, FVectorSupport{TSpace}} end

"""
    MPointvariate

Structure that subtypes `VariateForm`, indicating that a single sample
is a point on a manifold.
"""
struct MPointvariate <: VariateForm end

"""
    MPointSupport(M::AbstractManifold)

Value support for manifold-valued distributions (values from given
[`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)  `M`).
"""
struct MPointSupport{TM <: AbstractManifold} <: ValueSupport
    manifold::TM
end

"""
    MPointDistribution{TM<:AbstractManifold}

An abstract distribution for points on manifold of type `TM`.
"""
abstract type MPointDistribution{TM <: AbstractManifold} <:
Distribution{MPointvariate, MPointSupport{TM}} end

"""
    support(d::FVectorDistribution)

Get the object of type `FVectorSupport` for the distribution `d`.
"""
function Distributions.support(::FVectorDistribution) end

"""
    uniform_distribution(M::Grassmann{<:Any,ℝ}, p)

Uniform distribution on given (real-valued) [`Grassmann`](@ref) `M`.
Specifically, this is the normalized Haar measure on `M`.
Generated points will be of similar type as `p`.

The implementation is based on Section 2.5.1 in [Chikuse:2003](@cite);
see also Theorem 2.2.2(iii) in [Chikuse:2003](@cite).
"""
function uniform_distribution(M::Grassmann{ℝ}, p)
    n, k = get_parameter(M.size)
    μ = Distributions.Zeros(n, k)
    σ = one(eltype(p))
    Σ1 = Distributions.PDMats.ScalMat(n, σ)
    Σ2 = Distributions.PDMats.ScalMat(k, σ)
    d = MatrixNormal(μ, Σ1, Σ2)

    return ProjectedPointDistribution(M, d, (M, q, p) -> (q .= svd(p).U), p)
end

"""
    uniform_distribution(M::ProjectiveSpace{<:Any,ℝ}, p)

Uniform distribution on given [`ProjectiveSpace`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::ProjectiveSpace{ℝ}, p)
    d = Distributions.MvNormal(zero(p), 1.0 * I)
    return ProjectedPointDistribution(M, d, project!, p)
end

"""
    uniform_distribution(M::Stiefel{ℝ}, p)

Uniform distribution on given (real-valued) [`Stiefel`](@ref) `M`.
Specifically, this is the normalized Haar and Hausdorff measure on `M`.
Generated points will be of similar type as `p`.

The implementation is based on Section 2.5.1 in [Chikuse:2003](@cite);
see also Theorem 2.2.1(iii) in [Chikuse:2003](@cite).
"""
function uniform_distribution(M::Stiefel{ℝ}, p)
    n, k = get_parameter(M.size)
    μ = Distributions.Zeros(n, k)
    σ = one(eltype(p))
    Σ1 = Distributions.PDMats.ScalMat(n, σ)
    Σ2 = Distributions.PDMats.ScalMat(k, σ)
    d = MatrixNormal(μ, Σ1, Σ2)

    return ProjectedPointDistribution(M, d, project!, p)
end

"""
    uniform_distribution(M::Sphere{n,ℝ}, p) where {n}

Uniform distribution on given [`Sphere`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::Sphere{ℝ}, p)
    n = get_parameter(M.size)[1]
    d = Distributions.MvNormal(zero(p), 1.0 * I)
    return ProjectedPointDistribution(M, d, project!, p)
end
