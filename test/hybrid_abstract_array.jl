using Manifolds
using Manifolds: HybridAbstractArray, HybridAbstractVector, HybridAbstractMatrix
using StaticArrays
using Test

@testset "HybridAbstractArray" begin
    @testset "Inner Constructors" begin
        @test HybridAbstractArray{Tuple{2}, Int, 1, 1, Vector{Int}}((3, 4)).data == [3, 4]
        @test HybridAbstractArray{Tuple{2}, Int, 1}([3, 4]).data == [3, 4]
        @test HybridAbstractArray{Tuple{2, 2}, Int, 2}(collect(3:6)).data == collect(3:6)
        @test size(HybridAbstractArray{Tuple{4, 5}, Int, 2}(undef).data) == (4, 5)
        @test size(HybridAbstractArray{Tuple{4, 5}, Int}(undef).data) == (4, 5)

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1}([2 3])

        # Bad parameters
        @test_throws Exception HybridAbstractArray{Tuple{1},Int,2}(undef)
        @test_throws Exception SArray{Tuple{3, 4},Int,1}(undef)

        # Parameter/input size mismatch
        @test_throws Exception HybridAbstractArray{Tuple{1},Int,2}([2; 3])
        @test_throws Exception HybridAbstractArray{Tuple{1},Int,2}((2, 3))
    end

    @testset "Outer Constructors" begin
        # From Array
        @test @inferred(HybridAbstractArray{Tuple{2},Float64,1}([1,2]))::HybridAbstractArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(HybridAbstractArray{Tuple{2},Float64}([1,2]))::HybridAbstractArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(HybridAbstractArray{Tuple{2}}([1,2]))::HybridAbstractArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(HybridAbstractArray{Tuple{2,2}}([1 2;3 4]))::HybridAbstractArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
        # From Array, reshaped
        @test_broken @inferred(HybridAbstractArray{Tuple{2,2}}([1,2,3,4]))::HybridAbstractArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        # Uninitialized
        @test @inferred(HybridAbstractArray{Tuple{2,2},Int,2}(undef)) isa HybridAbstractArray{Tuple{2,2},Int,2,2}
        @test @inferred(HybridAbstractArray{Tuple{2,2},Int}(undef)) isa HybridAbstractArray{Tuple{2,2},Int,2,2}

        # From Tuple
        @test @inferred(HybridAbstractArray{Tuple{2},Float64,1,1,Vector{Float64}}((1,2)))::HybridAbstractArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(HybridAbstractArray{Tuple{2},Float64}((1,2)))::HybridAbstractArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(HybridAbstractArray{Tuple{2}}((1,2)))::HybridAbstractArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(HybridAbstractArray{Tuple{2,2}}((1,2,3,4)))::HybridAbstractArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
    end

    @testset "HybridAbstractVector and HybridAbstractMatrix" begin
        @test @inferred(HybridAbstractVector{2}([1,2]))::HybridAbstractArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(HybridAbstractVector{2}((1,2)))::HybridAbstractArray{Tuple{2},Int,1,1} == [1,2]
        # Reshaping
        @test @inferred(HybridAbstractVector{2}([1 2]))::HybridAbstractArray{Tuple{2},Int,1,2} == [1,2]
        # Back to Vector
        @test Vector(HybridAbstractVector{2}((1,2))) == [1,2]
        @test convert(Vector, HybridAbstractVector{2}((1,2))) == [1,2]

        @test @inferred(HybridAbstractMatrix{2,2}([1 2; 3 4]))::HybridAbstractArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
        # Reshaping
        @test_broken @inferred(HybridAbstractMatrix{2,2}([1,2,3,4]))::HybridAbstractArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        @test @inferred(HybridAbstractMatrix{2,2}((1,2,3,4)))::HybridAbstractArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
        # Back to Matrix
        @test Matrix(HybridAbstractMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
        @test convert(Matrix, HybridAbstractMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
    end

    # setindex
    sa = HybridAbstractArray{Tuple{2}, Int, 1}([3, 4])
    sa[1] = 2
    @test sa.data == [2, 4]

    @testset "aliasing" begin
        a1 = rand(4)
        a2 = copy(a1)
        sa1 = HybridAbstractVector{4}(a1)
        sa2 = HybridAbstractVector{4}(a2)
        @test Base.mightalias(a1, sa1)
        @test Base.mightalias(sa1, HybridAbstractVector{4}(a1))
        @test !Base.mightalias(a2, sa1)
        @test !Base.mightalias(sa1, HybridAbstractVector{4}(a2))
        @test Base.mightalias(sa1, view(sa1, 1:2))
        @test Base.mightalias(a1, view(sa1, 1:2))
        @test Base.mightalias(sa1, view(a1, 1:2))
    end

    @testset "back to Array" begin
        @test Array(HybridAbstractArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int}(HybridAbstractArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int, 1}(HybridAbstractArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Vector(HybridAbstractArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test convert(Vector, HybridAbstractArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test Matrix(SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Matrix, SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Array, HybridAbstractArray{Tuple{2,2,2,2}, Int}(ones(2,2,2,2))) == ones(2,2,2,2)
        # Conversion after reshaping
        @test Array(HybridAbstractMatrix{2,2,Int,1,Vector{Int}}([1,2,3,4])) == [1 3; 2 4]
    end

    @testset "promotion" begin
        @test @inferred(promote_type(HybridAbstractVector{1,Float64,1,Vector{Float64}}, HybridAbstractVector{1,BigFloat,1,Vector{BigFloat}})) == HybridAbstractVector{1,BigFloat,1,Vector{BigFloat}}
        @test @inferred(promote_type(HybridAbstractVector{2,Int,1,Vector{Int}}, HybridAbstractVector{2,Float64,1,Vector{Float64}})) === HybridAbstractVector{2,Float64,1,Vector{Float64}}
        @test @inferred(promote_type(HybridAbstractMatrix{2,3,Float32,2,Matrix{Float32}}, HybridAbstractMatrix{2,3,Complex{Float64},2,Matrix{Complex{Float64}}})) === HybridAbstractMatrix{2,3,Complex{Float64},2,Matrix{Complex{Float64}}}
    end

    @testset "dynamically sized axes" begin
        A = rand(Int, 2, 3, 4)
        B = HybridAbstractArray{Tuple{2,3,StaticArrays.Dynamic()}, Int, 3}(A)
        @test size(B) == size(A)
        @test axes(B) == (SOneTo(2), SOneTo(3), axes(A, 3))
        @test axes(B, 1) == SOneTo(2)
        @test axes(B, 2) == SOneTo(3)

        @test B[1,2,3] == A[1,2,3]
        @test B[1,:,:] == A[1,:,:]
        inds = @SVector [2, 1]
        @test B[1,inds,:] == A[1,inds,:]
        @test B[:,:,2] == A[:,:,2]
        @test B[:,:,@SVector [2, 3]] == A[:,:,[2, 3]]

        B[1,2,3] = 42
        @test B[1,2,3] == 42
        B[:,2,3] = @SVector [10, 11]
        @test B[:,2,3] == @SVector [10, 11]
        B[:,:,1] = @SMatrix [1 2 3; 4 5 6]
        @test B[:,:,1] == @SMatrix [1 2 3; 4 5 6]
        B[1,2,:] = [10, 11, 12, 13]
    end
end
