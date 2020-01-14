@doc doc"""
    SymmetricMatrices{n,F} <: Manifold

The [`Manifold`](@ref) $ \operatorname{Sym} (n)$ consisting of the real- or complex-valued
symmetric matrices of size $ n\times n$, i.e. the set

````math
\operatorname{Sym}(n) = \bigl\{A \in \mathbb F^{n\times n} \big| A^{\mathrm{H}} = A \bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed
and the field $\mathbb F \in \{ \mathbb R, \mathbb C\}$ is set by the
[`AbstractNumbers`](@ref) `F`.

Though it is slighty redundant, usually the matrices are safed as $n\times n$ arrays.

# Constructor

    SymmetricMatrices(n::Int, F::AbstractNumbers=ℝ)

Generate the manifold of $n\times n$ symmetric metrices.
"""
struct SymmetricMatrices{n,F} <: Manifold end
SymmetricMatrices(n::Int,F::AbstractNumbers=ℝ) = SymmetricMatrices{n,F}()

@doc doc"""
    check_manifold_point(M::SymmetricMatrices{n,F}, x; kwargs...)

Check whether `x` is a valid manifold point on the [`SymmetricMatrices`](@ref) `M`, i.e.
whether `x` is a symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@ref) `F`.

The tolerance for the symmetry of `x` can be set using `kwargs...`.
"""
function check_manifold_point(M::SymmetricMatrices{n,F},x; kwargs...) where {n,F}
    if (F===ℝ) && !(eltype(x) <: Real)
        return DomainError(eltype(x),"The matrix $(x) does not lie on $M, since its values are not real.")
    end
    if (F===ℂ) && !(eltype(x) <: Real) && !(eltype(x) <: Complex)
        return DomainError(eltype(x),"The matrix $(x) does not lie on $M, since its values are not complex.")
    end
    if size(x) != (n,n)
        return DomainError(size(x),"The point $(x) does not lie on $M since its size ($(size(x))) does not match the representation size ($(representation_size(M))).")
    end
    if !isapprox(norm(x-transpose(x)),0.; kwargs...)
        return DomainError(norm(x-transpose(x)), "The point $(x) does not lie on $M, since it is not symmetric.")
    end
    return nothing
end

"""
    check_tangent_vector(M::SymmetricMatrices{n,F}, x, v; kwargs... )

Check whether `v` is a tangent vector to manifold point `x` on the
[`SymmetricMatrices`](@ref) `M`, i.e. `v` has to be a symmetric matrix of dimension `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@ref).

The tolerance for the symmetry of `x` and `v` can be set using `kwargs...`.
"""
function check_tangent_vector(M::SymmetricMatrices{n,F}, x, v; kwargs...) where {n,F}
    if (check_manifold_point(M, x;kwargs...) !== nothing)
        return check_manifold_point(M,x;kwargs...)
    end
    if (F===ℝ) && !(eltype(v) <: Real)
        return DomainError(eltype(v),"The matrix $(v) is not a tangent to a point on $M, since its values are not real.")
    end
    if (F===ℂ) && !(eltype(v) <: Real) && !(eltype(v) <: Complex)
        return DomainError(eltype(v),"The matrix $(v) is not a tangent to a point on $M, since its values are not complex.")
    end
    if size(v) != (n,n)
        return DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(M) since its size ($(size(v))) does not match the representation size ($(representation_size(M))).")
    end
    if !isapprox(norm(v-transpose(v)),0.; kwargs...)
        return DomainError(norm(v-transpose(v)),
            "The vector $(v) is not a tangent vector to $(x) on $(M), since it is not symmetric.")
    end
    return nothing
end

@doc doc"""
    distance(M::SymmetricMatrices, x, y)

Computeby using the inherited metric, i.e. taking the Frobenius-norm of the difference.

"""
distance(M::SymmetricMatrices, x, y) = norm(x-y)

@doc doc"""
    exp!(M::SymmetricMatrices, y, x, v)

Compute the exponential map eminating from `x` in tangent direction `v` on the
[`SymmetricMatrices`](@ref) `M`, which reads

````math
\exp_xv = x + v.
````
"""
exp(::SymmetricMatrices, ::Any...)
function exp!(M::SymmetricMatrices, y, x, v)
    y .= x .+ v
    return y
end


@doc doc"""
    flat(M::SymmetricMatrices, x, w::FVector{TangentSpaceType})

Compute the [`flat`](@ref flat(M::Manifold, x, w::FVector)) isomorphism of the
[`SymmetricMatrices`](@ref) `M` on the manifold point `x` and tangent vector `w`.

Since `M` is already a vector space over $\mathbb R$, this returns just the vector `w`.
"""
flat(::SymmetricMatrices, ::Any...)
function flat!(M::SymmetricMatrices, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

function get_coordinates(M::SymmetricMatrices{N, ℝ}, x, v, B::ArbitraryOrthonormalBasis{ℝ}) where N
    dim = manifold_dimension(M)
    vout = similar(v, dim)
    k = 1
    for i in 1:N, j in i:N
        scale = ifelse(i==j, 1, sqrt(2))
        vout[k] = v[i,j]*scale
        k += 1
    end
    return vout
end

function get_coordinates(M::SymmetricMatrices{N, ℂ}, x, v, B::ArbitraryOrthonormalBasis{ℝ}) where N
    dim = manifold_dimension(M)
    vout = similar(v, dim)
    k = 1
    for i in 1:N, j in i:N
        scale = ifelse(i==j, 1, sqrt(2))
        vout[k] = real(v[i,j]) * scale
        k += 1
        vout[k] = imag(v[i,j]) * scale
        k += 1
    end
    return vout
end

function get_vector(M::SymmetricMatrices{N, ℝ}, x, v, B::ArbitraryOrthonormalBasis{ℝ}) where N
    dim = manifold_dimension(M)
    vout = similar_result(M, get_vector, x)
    k = 1
    for i in 1:N, j in i:N
        scale = ifelse(i==j, 1, 1/sqrt(2))
        vout[i,j] = v[k]*scale
        vout[j,i] = v[k]*scale
        k += 1
    end
    return vout
end

function get_vector(M::SymmetricMatrices{N, ℂ}, x, v, B::ArbitraryOrthonormalBasis{ℝ}) where N
    dim = manifold_dimension(M)
    vout = similar_result(M, get_vector, x, x .* 1im)
    k = 1
    for i in 1:N, j in i:N
        scale = ifelse(i==j, 1, 1/sqrt(2))
        vout[i,j] = Complex(v[k], v[k+1])*scale
        vout[j,i] = vout[i,j]
        k += 2
    end
    return vout
end

@doc doc"""
    inner(M::SymmetricMatrices, x, w, v)

Compute the inner product of the two tangent vectors `w`, `v` from the tangent
space at `x` on the [`SymmetricMatrices`](@ref) `M` using the restriction of the
metric from the embedding, i.e.
````math
(v,w)_x = \operatorname{tr}(v^{\mathrm{H}}w),
````
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
@inline inner(M::SymmetricMatrices, x, w, v) = dot(w, v)

@doc doc"""
    log(M::SymmetricMatrices, x, y)
Compute the logarithmic map from `x` to `y` on the [`SymmetricMatrices`](@ref) `M`, which
reads

````math
\log_xy = y-x.
````
"""
log(::SymmetricMatrices, ::Any...)
function log!(M::SymmetricMatrices, v, x, y)
    v .= y.-x
    return v
end

@doc doc"""
manifold_dimension(M::SymmetricMatrices{n,𝔽})

Return the dimension of the [`SymmetricMatrices`](@ref) matrix `M` over the number system
`𝔽`, i.e.

````math
\dim \operatorname{Sym}(n,𝔽) = \frac{n(n+1)}{2} \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
function manifold_dimension(::SymmetricMatrices{N,𝔽}) where {N,𝔽}
    return div(N*(N+1),2)*real_dimension(𝔽)
end

@doc doc"""
    norm(M::SymmetricMatrices, x, v)

Compute the norm of the tangent vector `v` from the tangent space at `x` on the
[`SymmetricMatrices`](@ref) `M`, which is the norm from the embedding, i.e.

````math
\lVert v \rVert_x = \lVert v \rVert_2
````
"""
norm(M::SymmetricMatrices, x, v) = norm(v)

@doc doc"""
    project_point(M::SymmetricMatrices,x)

Projects `x` from the embedding onto the [`SymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\operatorname{Sym}(n)}(x) = \frac{1}{2} \bigl( x + x^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project_point(::SymmetricMatrices, ::Any...)
project_point!(M::SymmetricMatrices, x) = (x .= (x + transpose(x))./2)

@doc doc"""
    project_tangent(M::SymmetricMatrices, x, v)

Project the matrix `v` onto the tangent space at `x` on the [`SymmetricMatrices`](@ref) `M`,

````math
\operatorname{proj}_x(v) = \frac{1}{2} \bigl( v + v^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project_tangent(::SymmetricMatrices, ::Any...)
project_tangent!(M::SymmetricMatrices, w, x, v) = (w .= (v .+ transpose(v))./2 )

@doc doc"""
    representation_size(M::SymmetricMatrices)

Returns the size points on the [`SymmetricMatrices`](@ref) `M` are represented as, i.e.
for the $n\times n$ it's `(n,n)`.
"""
representation_size(::SymmetricMatrices{N}) where {N} = (N,N)


@doc doc"""
    sharp(M::SymmetricMatrices, x, w::FVector{CotangentSpaceType})

Compute the [`sharp`](@ref sharp(M::Manifold, x, w::FVector)) isomorphism of the
[`SymmetricMatrices`](@ref) `M` on the manifold point `x` and cotangent vector `w`.

Since `M` is already a vector space over $\mathbb R$, this returns just the vector `w`.
"""
sharp(::SymmetricMatrices, ::Any...)
function sharp!(M::SymmetricMatrices, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

@doc doc"""
    vector_transport_to(M::SymmetricMatrices, x, v, y, ::ParallelTransport)

Compute the parallel
[`vector_transport_to`](@ref vector_transport_to(M::Manifold, x, v, y, ParallelTransport()))
of `v` from the tangent space at `x` on the [`SymmetricMatrices`](@ref) `M` to `y`.
Since the metric is inherited from the embedding space, this is just the identity, i.e.

````math
P_{y\gets x}(v) = v.
````
"""
vector_transport_to(::SymmetricMatrices, ::Any...)
function vector_transport_to!(M::SymmetricMatrices, vto, x, v, y, ::ParallelTransport)
    copyto!(vto,v)
    return vto
end

@doc doc"""
    zero_tangent_vector(M, x)

Return the zero tangent vector for the tangent space at `x` on the
[`SymmetricMatrices`](@ref) `M`, i.e. the zero matrix.
"""
zero_tangent_vector(::SymmetricMatrices,::Any...)
function zero_tangent_vector!(M::SymmetricMatrices, v, x)
    fill!(v, 0)
    return v
end
