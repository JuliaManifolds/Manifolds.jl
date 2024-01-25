#
# This file requires Symplectic to be defined, since it need the symplectic inverse A^+
# This type is used in the Symplectic Grassmann
@doc raw"""
    Hamiltonian{T,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}

A type to store a Hamiltonian matrix, that is a square matrix for which ``A^+ = -A`` where

```math
A^+ = J_{2n}A^{\mathrm{T}}J_{2n}, \qquad J_{2n} \begin{pmatrix} 0 & I_n\\-I_n & 0 \end{pmatrix},
```

and ``I_n`` denotes the ``n×n``
"""
struct Hamiltonian{T,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}
    value::S
    function Hamiltonian(A::S) where {T,S<:AbstractMatrix{<:T}}
        n = div(size(A, 1), 2)
        @assert size(A, 1) == 2 * n "The first dimension of A ($(size(A,1))) is not even"
        @assert size(A, 2) == 2 * n "The matrix A is of size ($(size(A))), which is not square."
        return new{T,S}(A)
    end
end
# Avoid double wrapping / unwrap if that happened
Hamiltonian(A::Hamiltonian) = Hamiltonian(A.value)
# Conversion
function Matrix(A::Hamiltonian)
    return Matrix(A.value)
end

Base.:*(H::Hamiltonian, K::Hamiltonian) = Hamiltonian(H.value * K.value)
Base.:+(H::Hamiltonian, K::Hamiltonian) = Hamiltonian(H.value .+ K.value)
Base.:-(H::Hamiltonian, K::Hamiltonian) = Hamiltonian(H.value .- K.value)

function show(io::IO, A::Hamiltonian)
    return print(io, "Hamiltonian($(A.value))")
end
size(A::Hamiltonian) = size(A.value)

@doc raw"""
    HamiltonianMatrices{T,𝔽} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)
consisting of (real-valued) hamiltonian matrices of size ``n×n``, i.e. the set

````math
\mathfrak{sp}(2n,𝔽) = \bigl\{p  ∈ 𝔽^{2n×2n}\ \big|\ p^+ = p \bigr\},
````
where ``⋅^{+}`` denotes the [`symplectic_inverse`](@ref), and ``𝔽 ∈ \{ ℝ, ℂ\}``.

Though it is slightly redundant, usually the matrices are stored as ``2n×2n`` arrays.

The symbol ``\mathfak{sp}`` refers to the main usage within `Manifolds.jl` that is the
Lie algebra to the [`SymplecticMatrices`](@ref) interpreted as a Lie group with the
matrix multiplication as group operation.

# Constructor

    HamiltonianMatrices(2n::Int, field::AbstractNumbers=ℝ)

Generate the manifold of ``2n×2n`` Hamiltonian matrices.
"""
struct HamiltonianMatrices{T,𝔽} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function HamiltonianMatrices(n::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:type)
    n % 2 == 0 || throw(ArgumentError("The dimension of the symplectic manifold
                        embedding space must be even. Was odd, n % 2 == $(n % 2)."))
    size = wrap_type_parameter(parameter, (div(n, 2),))
    return HamiltonianMatrices{typeof(size),field}(size)
end

function active_traits(f, ::HamiltonianMatrices, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end

ManifoldsBase.@default_manifold_fallbacks HamiltonianMatrices Hamiltonian Hamiltonian value value

@doc raw"""
    ^(A::Hamilonian, ::typeof(+))

Compute the [`symplectic_inverse`](@ref) of a Hamiltonian (A)
"""
function ^(A::Hamiltonian, ::typeof(+))
    return Hamiltonian(symplectic_inverse(A.value))
end
function symplectic_inverse(A::Hamiltonian)
    return Hamiltonian(symplectic_inverse(A.value))
end

@doc raw"""
    check_point(M::HamiltonianMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`HamiltonianMatrices`](@ref) `M`, i.e.
whether `p` [`is_hamiltonian`](@ref).

The tolerance for the test of `p` can be set using `kwargs...`.
"""
function check_point(M::HamiltonianMatrices, p; kwargs...)
    if !is_hamiltonian(p; kwargs...)
        return DomainError(
            norm((Hamiltonian(p)^+).value + p),
            "The point $(p) does not lie on $M, since it is not hamiltonian.",
        )
    end
    return nothing
end

"""
    check_vector(M::HamiltonianMatrices{n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`HamiltonianMatrices`](@ref) `M`, i.e. `X` has to be a Hamiltonian matrix
The tolerance for [`is_hamiltonian`](@ref) `X` can be set using `kwargs...`.
"""
function check_vector(M::HamiltonianMatrices, p, X; kwargs...)
    if !is_hamiltonian(X; kwargs...)
        return DomainError(
            norm((Hamiltonian(X)^+).value + X),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not hamiltonian.",
        )
    end
    return nothing
end

embed(::HamiltonianMatrices, p) = p
embed(::HamiltonianMatrices, p, X) = X

function get_embedding(::HamiltonianMatrices{TypeParameter{Tuple{N}},𝔽}) where {N,𝔽}
    return Euclidean(2 * N, 2 * N; field=𝔽)
end
function get_embedding(M::HamiltonianMatrices{Tuple{Int},𝔽}) where {𝔽}
    N = get_parameter(M.size)[1]
    return Euclidean(2 * N, 2 * N; field=𝔽, parameter=:field)
end

"""
    is_flat(::HamiltonianMatrices)

Return true. [`HamiltonianMatrices`](@ref) is a flat manifold.
"""
is_flat(M::HamiltonianMatrices) = true

@doc raw"""
    is_hamiltonian(A::AbstractMatrix; kwargs...)

Test whether a matrix `A` is hamiltonian.
The test consists of verifying whether

```math
A^+ = -A
```
where ``A^+`` denotes the [`symplectic_inverse`](@ref) of `A`.

The passed keyword arguments are passed on to `isapprox`
check within
"""
function is_hamiltonian(A::AbstractMatrix; kwargs...)
    return isapprox(symplectic_inverse(A), -A; kwargs...)
end
function is_hamiltonian(A::Hamiltonian; kwargs...)
    return isapprox((A^+).value, -A.value; kwargs...)
end

function Base.show(io::IO, ::HamiltonianMatrices{TypeParameter{Tuple{n}},F}) where {n,F}
    return print(io, "HamiltonianMatrices($(2n), $(F))")
end
function Base.show(io::IO, M::HamiltonianMatrices{Tuple{Int},F}) where {F}
    n = get_parameter(M.size)[1]
    return print(io, "HamiltonianMatrices($(2n), $(F); parameter=:field)")
end

@doc raw"""
    pX = rand(M::HamiltonianMatrices; σ::Real=1.0, vector_at=nothing)
    rand!(M::HamiltonianMatrices, pX; σ::Real=1.0, vector_at=nothing)

Generate a random Hamiltonian matrix. Since these are a submanifold of ``ℝ^{2n×2n}``,
the same method applies for points and tangent vectors.
This can also be done in-place of `pX`.

The construction is based on generating one normally-distributed
``n×n`` matrix ``A`` and two symmetric ``n×n`` matrices ``B, C`` which are then stacked:

```math
p = \begin{pmatrix} A & B\\ C & -A^{\mathrm{T}} \end{pmatrix}
```

"""
rand(M::HamiltonianMatrices; σ::Real=1.0)

function rand!(
    rng::AbstractRNG,
    M::HamiltonianMatrices{<:Any,ℝ},
    pX;
    σ::Real=one(real(eltype(pX))),
    vector_at=nothing,
)
    n = get_parameter(M.size)[1]
    p1 = @view(pX[1:n, 1:n])
    p2 = @view(pX[1:n, (n + 1):(2n)])
    p3 = @view(pX[(n + 1):(2n), 1:n])
    p4 = @view(pX[(n + 1):(2n), (n + 1):(2n)])
    randn!(rng, p1)
    p4 .= -p1'
    randn!(rng, p2)
    randn!(rng, p3)
    p2 .= (1 / 2) .* (p2 .+ p2')
    p3 .= (1 / 2) .* (p2 .+ p2')
    pX .*= σ
    return pX
end
