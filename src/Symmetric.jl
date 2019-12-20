@doc doc"""
    SymmetricMatrices{n,T} <: Manifold

The [`Manifold`](@ref) $ \operatorname{Sym} (n)$ consisting of the `T`-valued, symmetric matrices of size $ n\times n$, i.e. the set 

$\operatorname{Sym}(n) = \bigl\{A \in \mathbb F^{n\times n} \big| A^{\mathrm{T}} = A \bigr\}$

where $\mathbb F \in \{ \mathbb R, \mathbb C\}$.
For `T`, there is a distinction between 
* `T = Real`: real-valued matrix entries
* `T = Complex`: complex-valued matrix entries

Being slightly inefficient, the matrices are safed as $n\times n$ arrays despite the redundance of the lower triangle.


# Constructor

    SymmetricMatrices(n::Int,T::Type = Real)

"""
struct SymmetricMatrices{N,T} <: Manifold end
SymmetricMatrices(N::Int,T::Type = Real) = SymmetricMatrices{N,T}()

function representation_size(::SymmetricMatrices{N,T}) where {N,T}
    return (N,N)
end

@doc doc"""
    manifold_dimension(M::SymmetricMatrices{n,Real})

Return the dimension of the [`SymmetricMatrices`](@ref) matrix `M` with real-valued entries, i.e. 
````math
    \frac{n(n+1)}{2}
````
"""
manifold_dimension(M::SymmetricMatrices{N,Real}) where {N,T} = div(N*(N+1),2)

@doc doc"""
    manifold_dimension(M::SymmetricMatrices{n,Complex})

Return the dimension of the [`SymmetricMatrices`](@ref) matrix `M` with complex-valued entries, i.e. 
````math
    n(n+1)
````
"""
manifold_dimension(M::SymmetricMatrices{N,Complex}) where {N,T} = N*(N+1)

@doc doc"""
    project_point!(M::SymmetricMatrices,x)
    
Implements the [`project_point`](@ref project_point!(M::Manifold, x)) of an arbitrary matrix `x` onto the manifold of [`SymmetricMatrices`](@ref), i.e. 
````math
x_{new} = \frac{1}{2} \left( x_{old} + x_{old}^{\mathrm{T}} \right).$
````
The value of the variable `x` is overwritten.
"""
project_point!(M::SymmetricMatrices, x) = (x .= 1/2 * (x + transpose(x)))

@doc doc"""
    project_tangent!(M::SymmetricMatrices, w, x, v)
    
Implements the [`project_tangent!`](@ref project_tangent!(M::Manifold, w, x, v)) of the matrix `v` onto the tangent space of the manifold of [`SymmetricMatrices`](@ref),
i.e. the [`SymmetricMatrices`](@ref) itself. The result is stored in `w`:
````math
w = \frac{1}{2} \left( v + v^{\mathrm{T}} \right)
````

"""
project_tangent!(M::SymmetricMatrices, w, x, v) = (w .= 1/2 * (v + transpose(v)))


@doc doc"""
    inner(M::SymmetricMatrices, x, w, v)

compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the [`SymmetricMatrices`](@ref) `M` using the restriction of the
metric from the embedding, i.e. 
````math
(v,w)_x = \operatorname{tr}(v^{\mathrm{T}}w)
````
"""
@inline inner(M::SymmetricMatrices, x, w, v) = dot(w, v)


norm(M::SymmetricMatrices, x, v) = norm(v)

"""
    distance(M::SymmetricMatrices, x, y)

by using the inherited metric, i.e. taking the Frobenius-norm of the difference.

"""

distance(M::SymmetricMatrices, x, y) = norm(x-y)

@doc doc"""
    exp!(M::SymmetricMatrices, y, x, v)
Computes the exponential map [`exp!`](@ref exp!(M::Manifold, y, x, v, t=1)) of the tangent vector `v` at
manifold point `x` lying on the [`SymmetricMatrices`](@ref) manifold `M`. 
The result is saved in `y`. 
For the [`SymmetricMatrices`](@ref), the exponential map is simply the sum, i.e. 
````math
    y = x + v
````
"""
function exp!(M::SymmetricMatrices, y, x, v)
    y .= x + v
    return y
end

@doc doc"""
    log!(M::SymmetricMatrices, v, x, y)
Computes the logarithmic map from the manifold point `x` to the manifold point `y`, both lying on the [`SymmetricMatrices`](@ref) manifold `M`. 
The result is saved in `v`. 
For the [`SymmetricMatrices`](@ref), the logarithmic map is simply the difference, i.e. 
````math
    v = y-x
````
"""
function log!(M::SymmetricMatrices, v, x, y)
    v .= y-x
    return v
end

function zero_tangent_vector!(M::SymmetricMatrices, v, x)
    fill!(v, 0)
    return v
end


@doc doc"""
    vector_transport_to!(M::SymmetricMatrices, vto, x, v, y)
Computes the [`vector_transport_to!`](@ref vector_transport_to!(M::Manifold, vto, x, v, y, ProjectionTransport())) 
of the tangent vector `v` from the manifold point `x` on the [`SymmetricMatrices`](@ref) `M`
to the manifold point $y\in M$. The result is stored in `vto`.
Since the metric is inherited from the embedding space, this is just the identity.
"""
function vector_transport_to!(M::SymmetricMatrices, vto, x, v, y, ::ParallelTransport)
    vto .= v
    return vto
end

@doc doc"""
    flat!(M::SymmetricMatrices, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
Computes the [`flat!`](@ref flat!(M::Manifold, v::FVector, x, w::FVector)) isomorphism of the [`SymmetricMatrices`](@ref) manifold `M` 
on the manifold point `x` and tangent vector `w`. 
The result is stored in `v`. 

As `M` is already a vector space over $\mathbb R$, this returns just the vector `w`.
"""
function flat!(M::SymmetricMatrices, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

@doc doc"""
    sharp!(M::SymmetricMatrices, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
Computes the [`sharp!`](@ref sharp!(M::Manifold, v::FVector, x, w::FVector)) isomorphism of the [`SymmetricMatrices`](@ref) manifold `M` 
on the manifold point `x` and cotangent vector `w`. 
The result is stored in `v`. 

As `M` is already a vector space over $\mathbb R$, this returns just the vector `w`.
"""
function sharp!(M::SymmetricMatrices, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
end


@doc doc"""
    check_manifold_point(M::SymmetricMatrices,x; kwargs...)

Checks whether `x` is a valid manifold point on the [`SymmetricMatrices`](@ref) `M`, i.e. is a symmetric matrix
of size `(N,N)` with values of type `T`.

The tolerance for the symmetry of `x` can be set using `kwargs`.
"""
function check_manifold_point(M::SymmetricMatrices{N,T},x; kwargs...) where {N,T}
    if !(eltype(x) <: T)
        return DomainError(eltype(x),"The matrix $(x) does not lie on $M, since its values are not of type $T.")
    end
    if size(x) != (N,N)
        return DomainError(size(x),"The point $(x) does not lie on $M, since its size does not match manifold_dimension(M).")
    end
    if !isapprox(norm(x-transpose(x)),0.; kwargs...)
        return DomainError(norm(x-transpose(x)), "The point $(x) does not lie on $M, since it is not symmetric.")
    end
    return nothing
end

"""
    check_tangent_vector(M::SymmetricMatrices,x,v; kwargs... )

Checks whether `v` is a valid tangent vector to manifold point `x` on the [`SymmetricMatrices`](@ref) matrices `M`, i.e.
after [`is_manifold_point`](@ref)(M,x), `v` has to be a symmetric matrix of dimension `(N,N)`.

The tolerance for the symmetry of `x` can be set using `kwargs`.
"""
function check_tangent_vector(M::SymmetricMatrices{N,T},x,v; kwargs...) where {N,T}
    if (check_manifold_point(M,x;kwargs...) !== nothing)
        return check_manifold_point(M,x;kwargs...)
    end

    if !(eltype(v) <: T)
        return DomainError(eltype(v),"The matrix $(v) is not a tangent to a point on $M, since its values are not of type $T.")
    end
    if size(v) != (N,N)
        return DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(M) since its size does not match manifold_dimension(M).")
    end
    if !isapprox(norm(v-transpose(v)),0.; kwargs...)
        return DomainError(norm(v-transpose(v)),
            "The vector $(v) is not a tangent vector to $(x) on $(M), since it is not symmetric.")
    end
    return nothing
end

