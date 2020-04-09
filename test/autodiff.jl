using Test
using Manifolds: adbackend, adbackends, adbackend!, _gradient, _jacobian
using LinearAlgebra: Diagonal, dot

@testset "AD backend" begin
    @testset "adbackend" begin
        @test adbackend() == :finitedifferences
        @test adbackend(:default) == :finitedifferences
        @test adbackend(:finitedifferences) == :finitedifferences
        @test adbackends() == [:finitedifferences]

        @test adbackend!(:default) == :finitedifferences
        @test adbackend() == :finitedifferences
        @test adbackend!(:finitedifferences) == :finitedifferences
        @test adbackend() == :finitedifferences
        @test_throws ArgumentError adbackend!(:blah)
        @test_throws ArgumentError adbackend!(:forwarddiff)
    end

    using ForwardDiff

    @testset "forwarddiff" begin
        @test adbackend() == :finitedifferences
        @test adbackends() == [:finitedifferences, :forwarddiff]
        @test adbackend(:forwarddiff) == :forwarddiff

        @test adbackend!(:forwarddiff) == :forwarddiff
        @test adbackend() == :forwarddiff
        @test adbackend!(:finitedifferences) == :finitedifferences
        @test adbackend() == :finitedifferences

        adbackend!(:forwarddiff)
        @test adbackend!(:default) == :forwarddiff
        @test adbackend() == :forwarddiff
        adbackend!(:finitedifferences)
    end

    @testset "gradient/jacobian" begin
        adbackend!(:finitedifferences)
        @testset for backend in [:default, :finitedifferences, :forwarddiff]
            adbackend!(backend)
            @test _gradient(sin, 0.5) ≈ cos(0.5)
            @test _gradient(x -> dot(x, [1.0, 2.0, 3.0]), [4.0, 5.0, 6.0]) ≈ [1.0, 2.0, 3.0]
            J = _jacobian(x -> [x[1], x[2]^2, x[3]^3], [1.0, 2.0, 3.0])
            @test J ≈ collect(Diagonal([1.0, 4.0, 27.0])) atol = 1e-6
            J = _jacobian([1.0 2.0; 3.0 4.0]) do x
                return [x[1] x[3]^2; x[2]^2 x[4]^2]
            end
            J = _jacobian([1.0 2.0; 3.0 4.0]) do x
                return [x[1] x[2]^2; x[3]^2 x[4]^2]
            end
            @test J ≈ [1 0 0 0; 0 0 4 0; 0 6 0 0; 0 0 0 8] atol = 1e-6
        end
        adbackend!(:finitedifferences)
    end
end
