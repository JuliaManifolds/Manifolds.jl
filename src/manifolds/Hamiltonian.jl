#
# This file requires Symplectic to be defined, since it need the symplectic inverse A^+
# This type is used in the Symplectic Grassmann
@doc raw"""
    Hamiltonian{T,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}

A type to store an hamiltonien matrix, i.e. A square matrix matrix for which ``A^+ = -A`` where

```math
A^+ = J_{2n}A^{\mathrm{T}}J_{2n}, \qquad J_{2n} \begin{pmatrix} 0 & I_n\\-I_n & 0 \end{pmatrix},
```

and ``I_n`` denotes the ``n × n``
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
    return copy(A.value)
end

@doc raw"""
    ^(A::Hamilonian, ::typeof(+))

Compute the [`symplectic_inverse`](@ref) of a Hamiltonian (A)
"""
function ^(A::Hamiltonian, ::typeof(+))
    return Hamiltonian(symplectic_inverse(A.value))
end

@doc raw"""
    is_hamiltonian(A; kwargs...)

Test whether a matrix `A` is hamiltonian.
The test consists of verifying whether

```math
A^+ = -A
```
where ``A^+`` denotes the [`symplectic_inverse`](@ref) of `A`.

The passed keyword arguments are passed on to the [`isapprox`](@ref)
check within
"""
function is_hamiltonian(A::AbstractMatrix; kwargs...)
    return isapprox(symplectic_inverse(A), -A; kwargs...)
end
function is_hamiltonian(A::Hamiltonian; kwargs...)
    return isapprox((A^+).value, -A.value; kwargs...)
end

function show(io::IO, ::MIME"text/plain", A::Hamiltonian)
    return print(io, "Hamiltonian($(A.value))")
end

size(A::Hamiltonian) = size(A.value)
