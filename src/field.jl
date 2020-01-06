"""
    AbstractField

An abstract type to represent the field matrix manifolds are build upon, following the idea
of [TensorKit](https://github.com/Jutho/TensorKit.jl) in order to have concrete field types
to dispatch on. The two most common field types are [`RealNumbers`](@ref) (`ℝ` for short)
and [`ComplexNumbers`](@ref) (`ℂ`).
"""
abstract type AbstractField end

struct RealNumbers <: AbstractField end

struct ComplexNumbers <: AbstractField end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()

Base.show(io::IO, ::RealNumbers) = print(io, "ℝ")
Base.show(io::IO, ::ComplexNumbers) = print(io, "ℂ")

