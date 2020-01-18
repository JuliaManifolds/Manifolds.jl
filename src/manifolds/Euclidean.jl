@doc doc"""
    Euclidean{T<:Tuple} <: Manifold

Euclidean vector space $\mathbb R^n$.

# Constructor

    Euclidean(n)

Generate the $n$-dimensional vector space $\mathbb R^n$.

    Euclidean(n₁,n₂,...,nᵢ; field=ℝ)

Generate the vector space of $k=n_1n_2\cdot\ldots n_i$ values, i.e. the
$\mathbb F^{n_1, n_2, \ldots, n_d}$ whose
elements are interpreted as $n_1 \times,n_2\times\cdots\times n_i$ arrays.
For $d=2$ we obtain a matrix space.
The default `field=ℝ` can also be set to `field=ℂ`.
The dimension of this space is $k \dim_ℝ 𝔽$, where $\dim_ℝ 𝔽$ is the
[`real_dimension`](@ref) of the field $𝔽$.
"""
struct Euclidean{N<:Tuple,T} <: Manifold where {N,T<:AbstractNumbers} end

function Euclidean(n::Vararg{Int,N}; field::AbstractNumbers = ℝ) where {N}
    return Euclidean{Tuple{n...},field}()
end

"""
    EuclideanMetric <: RiemannianMetric

A general type for any manifold that employs the Euclidean Metric, for example
the [`Euclidean`](@ref) manifold itself, or the [`Sphere`](@ref), where every
tangent space (as a plane in the embedding) uses this metric (in the embedding).

Since the metric is independent of the field type, this metric is also used for
the Hermitian metrics, i.e. metrics that are analogous to the `EuclideanMetric`
but where the field type of the manifold is `ℂ`.

This metric is the default metric for example for the [`Euclidean`](@ref) manifold.
"""
struct EuclideanMetric <: RiemannianMetric end

det_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = one(eltype(x))

"""
    distance(M::Euclidean, x, y)

Compute the Euclidean distance between two points on the [`Euclidean`](@ref)
manifold `M`, i.e. for vectors it's just the norm of the difference, for matrices
and higher order arrays, the matrix and ternsor Frobenius norm, respectively.
"""
distance(::Euclidean, x, y) = norm(x .- y)

@doc doc"""
    exp(M::Euclidean, x, v)

Compute the exponential map on the [`Euclidean`](@ref) manifold `M` from `x` in direction
`v`, which in this case is just
````math
\exp_x v = x + v.
````
"""
exp(::Euclidean, ::Any...)

exp!(M::Euclidean, y, x, v) = (y .= x .+ v)

"""
    flat(M::Euclidean, x, w)

Transform a tangent vector into a cotangent. Since they can directly be identified in the
[`Euclidean`](@ref) case, this yields just the identity for a tangent vector `w` in the
tangent space of `x` on `M`. The result is returned also in place in `v`.
"""
flat(::Euclidean, ::Any...)

flat!(M::Euclidean, v::CoTFVector, x, w::TFVector) = copyto!(v, w)

function get_basis(M::Euclidean{<:Tuple,ℝ}, x, B::ArbitraryOrthonormalBasis)
    vecs = [_euclidean_basis_vector(x, i) for i in eachindex(x)]
    return PrecomputedOrthonormalBasis(vecs)
end
function get_basis(M::Euclidean{<:Tuple,ℂ}, x, B::ArbitraryOrthonormalBasis)
    vecs = [_euclidean_basis_vector(x, i) for i in eachindex(x)]
    return PrecomputedOrthonormalBasis([vecs; im * vecs])
end
function get_basis(M::Euclidean, x, B::DiagonalizingOrthonormalBasis)
    vecs = get_basis(M, x, ArbitraryOrthonormalBasis()).vectors
    kappas = zeros(real(eltype(x)), manifold_dimension(M))
    return PrecomputedDiagonalizingOrthonormalBasis(vecs, kappas)
end

function get_coordinates(M::Euclidean{<:Tuple,ℝ}, x, v, B::ArbitraryOrDiagonalizingBasis)
    S = representation_size(M)
    PS = prod(S)
    return reshape(v, PS)
end
function get_coordinates(M::Euclidean{<:Tuple,ℂ}, x, v, B::ArbitraryOrDiagonalizingBasis)
    S = representation_size(M)
    PS = prod(S)
    return [reshape(real(v), PS); reshape(imag(v), PS)]
end

function get_vector(M::Euclidean{<:Tuple,ℝ}, x, v, B::ArbitraryOrDiagonalizingBasis)
    S = representation_size(M)
    return reshape(v, S)
end
function get_vector(M::Euclidean{<:Tuple,ℂ}, x, v, B::ArbitraryOrDiagonalizingBasis)
    S = representation_size(M)
    N = div(length(v), 2)
    return reshape(v[1:N] + im * v[N+1:end], S)
end

function hat(M::Euclidean{N,ℝ}, x, vⁱ) where {N}
    return reshape(vⁱ, representation_size(TangentBundleFibers(M)))
end

hat!(::Euclidean{N,ℝ}, v, x, vⁱ) where {N} = copyto!(v, vⁱ)

@doc doc"""
    injectivity_radius(M::Euclidean)

Return the injectivity radius on the [`Euclidean`](@ref) `M`, which is $\infty$.
"""
injectivity_radius(::Euclidean) = Inf

@doc doc"""
    inner(M::Euclidean, x, v, w)

Compute the inner product on the [`Euclidean`](@ref) `M`, which is just
the inner product on the real-valued or complex valued vector space
of arrays (or tensors) of size $n_1\times n_2 \times \cdots \times n_i$, i.e.

````math
g_x(v,w) = \sum_{k\in I} \overline{v}_{k} w_{k},
````
where $I$ is the set of integer vectors $k\in\mathbb N^i$, such that for all
$1 \leq j \leq i$ it holds $1\leq k_j \leq n_j$.

For the special case of $i\leq 2$, i.e. matrices and vectors, this simplifies to
````math
g_x(v,w) = w^{\mathrm{H}}v,
````
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
inner(::Euclidean, ::Any...)
@inline inner(::Euclidean, x, v, w) = dot(v, w)
@inline inner(::MetricManifold{<:Manifold,EuclideanMetric}, x, v, w) = dot(v, w)

inverse_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = local_metric(M, x)

is_default_metric(::Euclidean, ::EuclideanMetric) = Val(true)

function local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, x)
    return Diagonal(ones(SVector{size(x, 1),eltype(x)}))
end

@doc doc"""
    log(M::Euclidean, x, y)

Compute the logarithmic map on the [`Euclidean`](@ref) `M` from `x` to `y`,
which in this case is just
````math
\log_x y = y - x.
````
"""
log(::Euclidean, ::Any...)

log!(M::Euclidean, v, x, y) = (v .= y .- x)

log_local_metric_density(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = zero(eltype(x))

@generated _product_of_dimensions(::Euclidean{N}) where {N} = prod(N.parameters)

"""
    manifold_dimension(M::Euclidean)

Return the manifold dimension of the [`Euclidean`](@ref) `M`, i.e.
the product of all array dimensions and the [`real_dimension`](@ref) of the
underlying number system.
"""
function manifold_dimension(M::Euclidean{N,𝔽}) where {N,𝔽}
    return _product_of_dimensions(M) * real_dimension(𝔽)
end

mean(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}; kwargs...) = mean(x)
function mean(
    ::Euclidean{Tuple{1}},
    x::AbstractVector{<:Number},
    w::AbstractWeights;
    kwargs...,
)
    return mean(x, w)
end
mean(::Euclidean, x::AbstractVector; kwargs...) = mean(x)

function mean!(M::Euclidean, y, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, y, x, w, GeodesicInterpolation(); kwargs...)
end

function mean_and_var(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}; kwargs...)
    m, v = mean_and_var(x; kwargs...)
    return m, sum(v)
end
function mean_and_var(
    ::Euclidean{Tuple{1}},
    x::AbstractVector{<:Number},
    w::AbstractWeights;
    corrected = false,
    kwargs...,
)
    m, v = mean_and_var(x, w; corrected = corrected, kwargs...)
    return m, sum(v)
end
function mean_and_var(M::Euclidean, x::AbstractVector, w::AbstractWeights; kwargs...)
    return mean_and_var(M, x, w, GeodesicInterpolation(); kwargs...)
end

median(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}; kwargs...) = median(x)
function median(
    ::Euclidean{Tuple{1}},
    x::AbstractVector{<:Number},
    w::AbstractWeights;
    kwargs...,
)
    return median(x, w)
end

function median!(::Euclidean{Tuple{1}}, y, x::AbstractVector; kwargs...)
    return copyto!(y, [median(vcat(x...))])
end
function median!(::Euclidean{Tuple{1}}, y, x::AbstractVector, w::AbstractWeights; kwargs...)
    return copyto!(y, [median(vcat(x...), w)])
end

@doc doc"""
    norm(M::Euclidean, x, v)

Compute the norm of a tangent vector `v` at `x` on the [`Euclidean`](@ref)
`M`, i.e. since every tangent space can be identified with `M` itself
in this case, just the (Frobenius) norm of `v`.
"""
norm(::Euclidean, x, v) = norm(v)
norm(::MetricManifold{<:Manifold,EuclideanMetric}, x, v) = norm(v)

"""
    normal_tvector_distribution(M::Euclidean, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(M::Euclidean{Tuple{N}}, x, σ) where {N}
    d = Distributions.MvNormal(zero(x), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), x, d, project_vector!, x)
end

@doc doc"""
    project_point(M::Euclidean, x)

Project an arbitrary point `x` onto the [`Euclidean`](@ref) `M`, which
is of course just the identity map.
"""
project_point(::Euclidean, ::Any...)

project_point!(M::Euclidean, x) = x

"""
    project_tangent(M::Euclidean, x, v)

Project an arbitrary vector `v` into the tangent space of a point `x` on the
[`Euclidean`](@ref) `M`, which is just the identity, since any tangent
space of `M` can be identified with all of `M`.
"""
project_tangent(::Euclidean, ::Any...)

project_tangent!(M::Euclidean, w, x, v) = copyto!(w, v)

"""
    projected_distribution(M::Euclidean, d, [x])

Wrap the standard distribution `d` into a manifold-valued distribution. Generated
points will be of similar type to `x`. By default, the type is not changed.
"""
function projected_distribution(M::Euclidean, d, x)
    return ProjectedPointDistribution(M, d, project_point!, x)
end
function projected_distribution(M::Euclidean, d)
    return ProjectedPointDistribution(M, d, project_point!, rand(d))
end

"""
    representation_size(M::Euclidean)

Return the array dimensions required to represent an element on the
[`Euclidean`](@ref) `M`, i.e. the vector of all array dimensions.
"""
representation_size(::Euclidean{N}) where {N} = size_to_tuple(N)

"""
    sharp(M::Euclidean, x, w)

since cotangent and tangent vectors can directly be identified in the [`Euclidean`](@ref)
case, this yields just the identity for a cotangent vector `w` in the tangent space
of `x` on `M`.
"""
sharp(::Euclidean, ::Any...)

sharp!(M::Euclidean, v::TFVector, x, w::CoTFVector) = copyto!(v, w)

"""
    vector_transport_to(M::Euclidean, x, v, y, ::ParallelTransport)

Parallely transport the vector `v` from the tangent space at `x` to the tangent space at `y`
on the [`Euclidean`](@ref) `M`, which simplifies to the identity.
"""
vector_transport_to(::Euclidean, ::Any, ::Any, ::Any, ::ParallelTransport)

vector_transport_to!(M::Euclidean, vto, x, v, y, ::ParallelTransport) = copyto!(vto, v)

var(::Euclidean, x::AbstractVector; kwargs...) = sum(var(x; kwargs...))
function var(::Euclidean, x::AbstractVector{T}, m::T; kwargs...) where {T}
    return sum(var(x; mean = m, kwargs...))
end

vee(::Euclidean{N,ℝ}, x, v) where {N} = vec(v)

vee!(::Euclidean{N,ℝ}, vⁱ, x, v) where {N} = copyto!(vⁱ, v)

"""
    zero_tangent_vector(M::Euclidean, x)

Return the zero vector in the tangent space of `x` on the [`Euclidean`](@ref)
`M`, which here is just a zero filled array the same size as `x`.
"""
zero_tangent_vector(::Euclidean, ::Any...)

zero_tangent_vector!(M::Euclidean, v, x) = fill!(v, 0)
