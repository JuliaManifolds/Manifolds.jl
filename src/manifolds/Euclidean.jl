@doc raw"""
    Euclidean{T<:Tuple} <: Manifold

Euclidean vector space $ℝ^n$.

# Constructor

    Euclidean(n)

Generate the $n$-dimensional vector space $ℝ^n$.

    Euclidean(n₁,n₂,...,nᵢ; field=ℝ)

Generate the vector space of $k=n_1n_2\cdot… n_i$ values, i.e. the
$𝔽^{n_1, n_2,…, n_d}$ whose
elements are interpreted as $n_1 ×,n_2 × … × n_i$ arrays.
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

^(M::Euclidean, n::Int) = ^(M, (n,))
function ^(::Euclidean{T,F}, n::NTuple{N,Int}) where {T,F,N}
    return Euclidean{Tuple{T.parameters...,n...},F}()
end

det_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, p) = one(eltype(p))

"""
    distance(M::Euclidean, x, y)

Compute the Euclidean distance between two points on the [`Euclidean`](@ref)
manifold `M`, i.e. for vectors it's just the norm of the difference, for matrices
and higher order arrays, the matrix and ternsor Frobenius norm, respectively.
"""
distance(::Euclidean, p, q) = norm(p .- q)

@doc raw"""
    exp(M::Euclidean, p, X)

Compute the exponential map on the [`Euclidean`](@ref) manifold `M` from `x` in direction
`X`, which in this case is just
````math
\exp_p X = p + X.
````
"""
exp(::Euclidean, ::Any...)

exp!(M::Euclidean, q, p, X) = (q .= p .+ X)

"""
    flat(M::Euclidean, p, X)

Transform a tangent vector `X` into a cotangent. Since they can directly be identified in the
[`Euclidean`](@ref) case, this yields just the identity for a tangent vector `w` in the
tangent space of `p` on `M`.
"""
flat(::Euclidean, ::Any...)

flat!(M::Euclidean, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

function get_basis(M::Euclidean{<:Tuple,ℝ}, p, B::ArbitraryOrthonormalBasis)
    vecs = [_euclidean_basis_vector(p, i) for i in eachindex(p)]
    return PrecomputedOrthonormalBasis(vecs)
end
function get_basis(M::Euclidean{<:Tuple,ℂ}, p, B::ArbitraryOrthonormalBasis)
    vecs = [_euclidean_basis_vector(p, i) for i in eachindex(p)]
    return PrecomputedOrthonormalBasis([vecs; im * vecs])
end
function get_basis(M::Euclidean, p, B::DiagonalizingOrthonormalBasis)
    vecs = get_basis(M, p, ArbitraryOrthonormalBasis()).vectors
    kappas = zeros(real(eltype(p)), manifold_dimension(M))
    return PrecomputedDiagonalizingOrthonormalBasis(vecs, kappas)
end

function get_coordinates(M::Euclidean{<:Tuple,ℝ}, p, X, B::ArbitraryOrDiagonalizingBasis)
    S = representation_size(M)
    PS = prod(S)
    return reshape(X, PS)
end
function get_coordinates(M::Euclidean{<:Tuple,ℂ}, p, X, B::ArbitraryOrDiagonalizingBasis)
    S = representation_size(M)
    PS = prod(S)
    return [reshape(real(X), PS); reshape(imag(X), PS)]
end

function get_vector(M::Euclidean{<:Tuple,ℝ}, p, X, B::ArbitraryOrDiagonalizingBasis)
    S = representation_size(M)
    return reshape(X, S)
end
function get_vector(M::Euclidean{<:Tuple,ℂ}, p, X, B::ArbitraryOrDiagonalizingBasis)
    S = representation_size(M)
    N = div(length(X), 2)
    return reshape(X[1:N] + im * X[N+1:end], S)
end

function hat(M::Euclidean{N,ℝ}, p, vⁱ) where {N}
    return reshape(vⁱ, representation_size(TangentBundleFibers(M)))
end

hat!(::Euclidean{N,ℝ}, v, p, vⁱ) where {N} = copyto!(v, vⁱ)

@doc raw"""
    injectivity_radius(M::Euclidean)

Return the injectivity radius on the [`Euclidean`](@ref) `M`, which is $∞$.
"""
injectivity_radius(::Euclidean) = Inf

@doc raw"""
    inner(M::Euclidean, p, X, Y)

Compute the inner product on the [`Euclidean`](@ref) `M`, which is just
the inner product on the real-valued or complex valued vector space
of arrays (or tensors) of size $n_1 × n_2  ×  …  × n_i$, i.e.

````math
g_p(X,Y) = \sum_{k ∈ I} \overline{X}_{k} Y_{k},
````
where $I$ is the set of integer vectors $k ∈ ℕ^i$, such that for all
$1 \leq j \leq i$ it holds $1\leq k_j \leq n_j$.

For the special case of $i\leq 2$, i.e. matrices and vectors, this simplifies to
````math
g_p(Y,Y) = X^{\mathrm{H}}Y,
````
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
inner(::Euclidean, ::Any...)
@inline inner(::Euclidean, p, X, Y) = dot(X, Y)
@inline inner(::MetricManifold{<:Manifold,EuclideanMetric}, p, X, Y) = dot(X, Y)

inverse_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, p) = local_metric(M, p)

is_default_metric(::Euclidean, ::EuclideanMetric) = Val(true)

function local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, p)
    return Diagonal(ones(SVector{size(p, 1),eltype(p)}))
end

@doc raw"""
    log(M::Euclidean, p, q)

Compute the logarithmic map on the [`Euclidean`](@ref) `M` from `p` to `q`,
which in this case is just
````math
\log_p q = q-p.
````
"""
log(::Euclidean, ::Any...)

log!(M::Euclidean, X, p, q) = (X .= q .- p)

log_local_metric_density(M::MetricManifold{<:Manifold,EuclideanMetric}, p) = zero(eltype(p))

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

function mean!(M::Euclidean, p, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, p, x, w, GeodesicInterpolation(); kwargs...)
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

function median!(::Euclidean{Tuple{1}}, p, x::AbstractVector; kwargs...)
    return copyto!(p, [median(vcat(x...))])
end
function median!(::Euclidean{Tuple{1}}, p, x::AbstractVector, w::AbstractWeights; kwargs...)
    return copyto!(p, [median(vcat(x...), w)])
end

@doc raw"""
    norm(M::Euclidean, p, X)

Compute the norm of a tangent vector `X` at `p` on the [`Euclidean`](@ref)
`M`, i.e. since every tangent space can be identified with `M` itself
in this case, just the (Frobenius) norm of `X`.
"""
norm(::Euclidean, p, X) = norm(X)
norm(::MetricManifold{<:Manifold,EuclideanMetric}, p, X) = norm(X)

"""
    normal_tvector_distribution(M::Euclidean, p, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `p`.
"""
function normal_tvector_distribution(M::Euclidean{Tuple{N}}, p, σ) where {N}
    d = Distributions.MvNormal(zero(p), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), p, d, project_vector!, p)
end

@doc raw"""
    project_point(M::Euclidean, p)

Project an arbitrary point `p` onto the [`Euclidean`](@ref) `M`, which
is of course just the identity map.
"""
project_point(::Euclidean, ::Any...)

project_point!(M::Euclidean, p) = p

"""
    project_tangent(M::Euclidean, p, X)

Project an arbitrary vector `X` into the tangent space of a point `p` on the
[`Euclidean`](@ref) `M`, which is just the identity, since any tangent
space of `M` can be identified with all of `M`.
"""
project_tangent(::Euclidean, ::Any...)

project_tangent!(M::Euclidean, Y, p, X) = copyto!(Y, X)

"""
    projected_distribution(M::Euclidean, d, [p])

Wrap the standard distribution `d` into a manifold-valued distribution. Generated
points will be of similar type to `p`. By default, the type is not changed.
"""
function projected_distribution(M::Euclidean, d, p)
    return ProjectedPointDistribution(M, d, project_point!, p)
end
function projected_distribution(M::Euclidean, d)
    return ProjectedPointDistribution(M, d, project_point!, rand(d))
end

"""
    representation_size(M::Euclidean)

Return the array dimensions required to represent an element on the
[`Euclidean`](@ref) `M`, i.e. the vector of all array dimensions.
"""
@generated representation_size(::Euclidean{N}) where {N} = size_to_tuple(N)

"""
    sharp(M::Euclidean, p, ξ)

Transform the cotangent vector `ξ` at `p` on the [`Euclidean`](@ref) `M` to a tangent vector `X`.
Since cotangent and tangent vectors can directly be identified in the [`Euclidean`](@ref)
case, this yields just the identity.
"""
sharp(::Euclidean, ::Any...)

sharp!(M::Euclidean, X::TFVector, x, ξ::CoTFVector) = copyto!(X, ξ)

function show(io::IO, ::Euclidean{N,F}) where {N,F}
    print(io, "Euclidean($(join(N.parameters, ", ")); field = $(F))")
end

"""
    vector_transport_to(M::Euclidean, p, X, q, ::ParallelTransport)

Parallely transport the vector `X` from the tangent space at `p` to the tangent space at `q`
on the [`Euclidean`](@ref) `M`, which simplifies to the identity.
"""
vector_transport_to(::Euclidean, ::Any, ::Any, ::Any, ::ParallelTransport)

vector_transport_to!(M::Euclidean, Y, p, X, q, ::ParallelTransport) = copyto!(Y, X)

var(::Euclidean, x::AbstractVector; kwargs...) = sum(var(x; kwargs...))
function var(::Euclidean, x::AbstractVector{T}, m::T; kwargs...) where {T}
    return sum(var(x; mean = m, kwargs...))
end

vee(::Euclidean{N,ℝ}, p, X) where {N} = vec(X)

vee!(::Euclidean{N,ℝ}, Xⁱ, p, X) where {N} = copyto!(Xⁱ, X)

"""
    zero_tangent_vector(M::Euclidean, x)

Return the zero vector in the tangent space of `x` on the [`Euclidean`](@ref)
`M`, which here is just a zero filled array the same size as `x`.
"""
zero_tangent_vector(::Euclidean, ::Any...)

zero_tangent_vector!(M::Euclidean, v, x) = fill!(v, 0)
