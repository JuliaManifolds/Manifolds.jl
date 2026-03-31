include("../header.jl")

using Quaternions

@testset "Special unitary matrices" begin
    @test manifold_volume(SpecialUnitaryMatrices(1)) ≈ 1
    @test manifold_volume(SpecialUnitaryMatrices(2)) ≈ 2 * π^2
    @test manifold_volume(SpecialUnitaryMatrices(3)) ≈ sqrt(3) * π^5
    @test manifold_volume(SpecialUnitaryMatrices(4)) ≈ sqrt(2) * 4 * π^9 / 12

    @test manifold_dimension(SpecialUnitaryMatrices(2)) == 3
    @test manifold_dimension(SpecialUnitaryMatrices(3)) == 8
    @test manifold_dimension(SpecialUnitaryMatrices(4)) == 15

    @test injectivity_radius(SpecialUnitaryMatrices(2)) == π * sqrt(2)
    @test injectivity_radius(SpecialUnitaryMatrices(3)) == π * sqrt(2)
    @test injectivity_radius(SpecialUnitaryMatrices(4)) == π * sqrt(2)
end

@testset "SO(4) and O(4) exp/log edge cases" begin
    Xs = [
        [0, 0, π, 0, 0, π],  # θ = (π, π)
        [0, 0, π, 0, 0, 0],  # θ = (π, 0)
        [0, 0, π / 2, 0, 0, π],  # θ = (π, π/2)
        [0, 0, π, 0, 0, 0] ./ 2,  # θ = (π/2, 0)
        [0, 0, π, 0, 0, π] ./ 2,  # θ = (π/2, π/2)
        [0, 0, 0, 0, 0, 0],  # θ = (0, 0)
        [0, 0, 1, 0, 0, 1] .* 1.0e-100, # α = β ≈ 0
        [0, 0, 1, 0, 0, 1] .* 1.0e-6, # α = β ⩰ 0
        [0, 0, 10, 0, 0, 1] .* 1.0e-6, # α ⪆ β ⩰ 0
        [0, 0, π / 4, 0, 0, π / 4 - 1.0e-6], # α ⪆ β > 0
    ]
    Ms = [Rotations(4), OrthogonalMatrices(4)]
    E = diagm(ones(4))
    for Xf in Xs
        for M in Ms
            @testset "for $Xf on $M" begin
                X = get_vector(M, E, Xf, DefaultOrthogonalBasis())
                p = exp(X)
                @test p ≈ exp(M, E, X)
                p3 = exp(M, E, log(M, E, p))
                # broken for 9 of the 10
                @test isapprox(M, p, p3; atol = 1.0e-4)
            end
        end
    end
    R1 = diagm([-1.0, -1.0, 1.0, 1.0])
    X1a = log(Rotations(4), E, R1)
    @test is_vector(Rotations(4), E, X1a)
    @test X1a[1, 2] ≈ π

    R2 = diagm([-1.0, 1.0, -1.0, 1.0])
    X2a = log(Rotations(4), E, R2)
    @test is_vector(Rotations(4), E, X2a)
    @test X2a[1, 3] ≈ π

    R3 = diagm([1.0, -1.0, -1.0, 1.0])
    X3a = log(Rotations(4), E, R3)
    @test is_vector(Rotations(4), E, X3a)
    @test X3a[2, 3] ≈ π
end
