
@doc raw"""
    const UnitaryMatrices{n,𝔽} = AbstarctUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices}

The manifold ``U(n,𝔽)`` of ``n×n`` complex matrices (when 𝔽=ℂ) or quaternionic matrices
(when 𝔽=ℍ) such that

``p^{\mathrm{H}}p = \mathrm{I}_n,``

where ``\mathrm{I}_n`` is the ``n×n`` identity matrix.
Such matrices `p` have a property that ``\lVert \det(p) \rVert = 1``.

The tangent spaces are given by

```math
    T_pU(n) \coloneqq \bigl\{
    X \big| pY \text{ where } Y \text{ is skew symmetric, i. e. } Y = -Y^{\mathrm{H}}
    \bigr\}
```

But note that tangent vectors are represented in the Lie algebra, i.e. just using ``Y`` in
the representation above.

# Constructor
    
    UnitaryMatrices(n, 𝔽::AbstractNumbers=ℂ)

see also [`OrthogonalMatrices`](@ref) for the real valued case.
"""
const UnitaryMatrices{n,𝔽} = GeneralUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices}

UnitaryMatrices(n::Int, 𝔽::AbstractNumbers=ℂ) = UnitaryMatrices{n,𝔽}()

check_size(::UnitaryMatrices{1,ℍ}, p::Number) = nothing
check_size(::UnitaryMatrices{1,ℍ}, p, X::Number) = nothing

embed(::UnitaryMatrices{1,ℍ}, p::Number) = SMatrix{1,1}(p)

embed(::UnitaryMatrices{1,ℍ}, p, X::Number) = SMatrix{1,1}(X)

function exp(::UnitaryMatrices{1,ℍ}, p, X::Number)
    return p * exp(X)
end

function get_coordinates_orthonormal(::UnitaryMatrices{1,ℍ}, p, X, ::QuaternionNumbers)
    return @SVector [X.v1, X.v2, X.v3]
end

function get_vector_orthonormal(
    ::UnitaryMatrices{1,ℍ},
    p::Quaternion,
    c,
    ::QuaternionNumbers,
)
    return Quaternion(0, c[1], c[2], c[3])
end

injectivity_radius(::UnitaryMatrices{1,ℍ}) = π

Base.isapprox(::UnitaryMatrices{1,ℍ}, x, y; kwargs...) = isapprox(x[], y[]; kwargs...)
Base.isapprox(::UnitaryMatrices{1,ℍ}, p, X, Y; kwargs...) = isapprox(X[], Y[]; kwargs...)

function log(::UnitaryMatrices{1,ℍ}, p::Number, q::Number)
    return log(conj(p) * q)
end

@doc raw"""
    manifold_dimension(M::UnitaryMatrices{n,ℂ}) where {n}
    
Return the dimension of the manifold unitary matrices.
```math
\dim_{\mathrm{U}(n)} = n^2.
```
"""
manifold_dimension(::UnitaryMatrices{n,ℂ}) where {n} = n^2
@doc raw"""
    manifold_dimension(M::UnitaryMatrices{n,ℍ})
    
Return the dimension of the manifold unitary matrices.
```math
\dim_{\mathrm{U}(n, ℍ)} = n(2n+1).
```
"""
manifold_dimension(::UnitaryMatrices{n,ℍ}) where {n} = n * (2n + 1)

Manifolds.number_of_coordinates(::UnitaryMatrices{1,ℍ}, ::AbstractBasis{ℍ}) = 3

project(::UnitaryMatrices{1,ℍ}, p) = normalize(p)

project(::UnitaryMatrices{1,ℍ}, p, X) = (X - conj(X)) / 2

function Random.rand(M::UnitaryMatrices{1,ℍ}; vector_at=nothing)
    if vector_at === nothing
        return normalize(quatrand())
    else
        project(M, vector_at, quatrand())
    end
end
function Random.rand(rng::AbstractRNG, M::UnitaryMatrices{1,ℍ}; vector_at=nothing)
    if vector_at === nothing
        return normalize(quatrand(rng))
    else
        project(M, vector_at, quatrand(rng))
    end
end

show(io::IO, ::UnitaryMatrices{n,ℂ}) where {n} = print(io, "UnitaryMatrices($(n))")
show(io::IO, ::UnitaryMatrices{n,ℍ}) where {n} = print(io, "UnitaryMatrices($(n), ℍ)")

Manifolds.zero_vector(::UnitaryMatrices{1,ℍ}, p) = zero(p)
