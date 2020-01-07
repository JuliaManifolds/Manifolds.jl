"""
    AbstractNumbers

An abstract type to represent the number system on which a manifold is built.

This provides concrete number types for dispatch. The two most common number types are
the fields [`RealNumbers`](@ref) (`ℝ` for short) and [`ComplexNumbers`](@ref) (`ℂ`).
"""
abstract type AbstractNumbers end

"""
    ℝ = RealNumbers()

The field of real numbers.
"""
struct RealNumbers <: AbstractNumbers end

"""
    ℂ = ComplexNumbers()

The field of complex numbers.
"""
struct ComplexNumbers <: AbstractNumbers end

"""
    ℍ = QuaternionNumbers()

The division algebra of quaternions.
"""
struct QuaternionNumbers <: AbstractNumbers end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()
const ℍ = QuaternionNumbers()

Base.show(io::IO, ::RealNumbers) = print(io, "ℝ")
Base.show(io::IO, ::ComplexNumbers) = print(io, "ℂ")
Base.show(io::IO, ::QuaternionNumbers) = print(io, "ℍ")

@doc doc"""
    real_dimension(𝔽::AbstractNumbers)

Return the real dimension $\dim_ℝ 𝔽$ of the [`AbstractNumbers`] system `𝔽`.
The real dimension is the dimension of a real vector space with which a number in `𝔽` can be
identified.
For example, [`ComplexNumbers`](@ref) have a real dimension of 2, and
[`QuaternionNumbers`](@ref) have a real dimension of 4.
"""
function real_dimension(𝔽::AbstractNumbers)
    error("real_dimension not defined for number system $(𝔽)")
end
real_dimension(::RealNumbers) = 1
real_dimension(::ComplexNumbers) = 2
real_dimension(::QuaternionNumbers) = 4
