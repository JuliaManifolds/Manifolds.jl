using Test
using Manifolds

struct NotImplementedNumbers <: Manifolds.AbstractNumbers end

@testset "Fields" begin
    @test_throws ErrorException real_dimension(NotImplementedNumbers())

    @test ℝ isa Manifolds.RealNumbers
    @test Manifolds.RealNumbers() === ℝ
    @test real_dimension(ℝ) == 1
    @test repr(ℝ) == "ℝ"
    @test ℝ^2 === Euclidean(2)
    @test ℝ^(2,3) === Euclidean(2, 3)

    @test ℂ isa Manifolds.ComplexNumbers
    @test Manifolds.ComplexNumbers() === ℂ
    @test real_dimension(ℂ) == 2
    @test repr(ℂ) == "ℂ"
    @test ℂ^2 === Euclidean(2; field = ℂ)
    @test ℂ^(2,3) === Euclidean(2, 3; field = ℂ)

    @test ℍ isa Manifolds.QuaternionNumbers
    @test Manifolds.QuaternionNumbers() === ℍ
    @test real_dimension(ℍ) == 4
    @test repr(ℍ) == "ℍ"
    @test ℍ^2 === Euclidean(2; field = ℍ)
    @test ℍ^(2,3) === Euclidean(2, 3; field = ℍ)
end
