"""
    AbstractNumbers

An abstract type to represent the field matrix manifolds are build upon, following the idea
of [TensorKit](https://github.com/Jutho/TensorKit.jl) in order to have concrete field types
to dispatch on. The two most common field types are [`RealNumbers`](@ref) (`ℝ` for short)
and [`ComplexNumbers`](@ref) (`ℂ`).
"""
abstract type AbstractNumbers end

"""
    ℝ = RealNumbers()

The [`field`](@ref AbstractNumbers) of real numbers.
"""
struct RealNumbers <: AbstractNumbers end

"""
    ℂ = ComplexNumbers()

The [`field`](@ref AbstractNumbers) of complex number.
"""
struct ComplexNumbers <: AbstractNumbers end

"""
    ℍ = QuaternionNumbers()

The [`field`](@ref AbstractNumbers) of quaternion numbers.
"""
struct QuaternionNumbers <: AbstractNumbers end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()
const ℍ = QuaternionNumbers()

Base.show(io::IO, ::RealNumbers) = print(io, "ℝ")
Base.show(io::IO, ::ComplexNumbers) = print(io, "ℂ")
Base.show(io::IO, ::QuaternionNumbers) = print(io, "ℍ")

"""
    field_dimension(𝔽::AbstractNumbers)

Return the dimension of the [`AbstractNumbers`](@ref).
The field dimension is the dimension of a real space with which a point on the field can be
identified. For example, [`ComplexNumbers`](@ref) have a field dimension of 2.
"""
field_dimension(𝔽::AbstractNumbers) = error("field_dimension not defined for field $(𝔽)")
field_dimension(::RealNumbers) = 1
field_dimension(::ComplexNumbers) = 2
field_dimension(::QuaternionNumbers) = 4
