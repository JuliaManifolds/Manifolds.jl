using Manifolds
using Manifolds: SizedAbstractArray, SizedAbstractVector, SizedAbstractMatrix
using StaticArrays
using Test

@testset "SizedAbstractArray" begin
    @testset "Inner Constructors" begin
        @test SizedAbstractArray{Tuple{2}, Int, 1, 1, Vector{Int}}((3, 4)).data == [3, 4]
        @test SizedAbstractArray{Tuple{2}, Int, 1}([3, 4]).data == [3, 4]
        @test SizedAbstractArray{Tuple{2, 2}, Int, 2}(collect(3:6)).data == collect(3:6)
        @test size(SizedAbstractArray{Tuple{4, 5}, Int, 2}(undef).data) == (4, 5)
        @test size(SizedAbstractArray{Tuple{4, 5}, Int}(undef).data) == (4, 5)

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1}([2 3])

        # Bad parameters
        @test_throws Exception SizedAbstractArray{Tuple{1},Int,2}(undef)
        @test_throws Exception SArray{Tuple{3, 4},Int,1}(undef)

        # Parameter/input size mismatch
        @test_throws Exception SizedAbstractArray{Tuple{1},Int,2}([2; 3])
        @test_throws Exception SizedAbstractArray{Tuple{1},Int,2}((2, 3))
    end

    @testset "Outer Constructors" begin
        # From Array
        @test @inferred(SizedAbstractArray{Tuple{2},Float64,1}([1,2]))::SizedAbstractArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedAbstractArray{Tuple{2},Float64}([1,2]))::SizedAbstractArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedAbstractArray{Tuple{2}}([1,2]))::SizedAbstractArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(SizedAbstractArray{Tuple{2,2}}([1 2;3 4]))::SizedAbstractArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
        # From Array, reshaped
        @test @inferred(SizedAbstractArray{Tuple{2,2}}([1,2,3,4]))::SizedAbstractArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        # Uninitialized
        @test @inferred(SizedAbstractArray{Tuple{2,2},Int,2}(undef)) isa SizedAbstractArray{Tuple{2,2},Int,2,2}
        @test @inferred(SizedAbstractArray{Tuple{2,2},Int}(undef)) isa SizedAbstractArray{Tuple{2,2},Int,2,2}

        # From Tuple
        @test @inferred(SizedAbstractArray{Tuple{2},Float64,1,1,Vector{Float64}}((1,2)))::SizedAbstractArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedAbstractArray{Tuple{2},Float64}((1,2)))::SizedAbstractArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedAbstractArray{Tuple{2}}((1,2)))::SizedAbstractArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(SizedAbstractArray{Tuple{2,2}}((1,2,3,4)))::SizedAbstractArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
    end

    @testset "SizedAbstractVector and SizedAbstractMatrix" begin
        @test @inferred(SizedAbstractVector{2}([1,2]))::SizedAbstractArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(SizedAbstractVector{2}((1,2)))::SizedAbstractArray{Tuple{2},Int,1,1} == [1,2]
        # Reshaping
        @test @inferred(SizedAbstractVector{2}([1 2]))::SizedAbstractArray{Tuple{2},Int,1,2} == [1,2]
        # Back to Vector
        @test Vector(SizedAbstractVector{2}((1,2))) == [1,2]
        @test convert(Vector, SizedAbstractVector{2}((1,2))) == [1,2]

        @test @inferred(SizedAbstractMatrix{2,2}([1 2; 3 4]))::SizedAbstractArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
        # Reshaping
        @test @inferred(SizedAbstractMatrix{2,2}([1,2,3,4]))::SizedAbstractArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        @test @inferred(SizedAbstractMatrix{2,2}((1,2,3,4)))::SizedAbstractArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
        # Back to Matrix
        @test Matrix(SizedAbstractMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
        @test convert(Matrix, SizedAbstractMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
    end

    # setindex
    sa = SizedAbstractArray{Tuple{2}, Int, 1}([3, 4])
    sa[1] = 2
    @test sa.data == [2, 4]

    @testset "aliasing" begin
        a1 = rand(4)
        a2 = copy(a1)
        sa1 = SizedAbstractVector{4}(a1)
        sa2 = SizedAbstractVector{4}(a2)
        @test Base.mightalias(a1, sa1)
        @test Base.mightalias(sa1, SizedAbstractVector{4}(a1))
        @test !Base.mightalias(a2, sa1)
        @test !Base.mightalias(sa1, SizedAbstractVector{4}(a2))
        @test Base.mightalias(sa1, view(sa1, 1:2))
        @test Base.mightalias(a1, view(sa1, 1:2))
        @test Base.mightalias(sa1, view(a1, 1:2))
    end

    @testset "back to Array" begin
        @test Array(SizedAbstractArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int}(SizedAbstractArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int, 1}(SizedAbstractArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Vector(SizedAbstractArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test convert(Vector, SizedAbstractArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test Matrix(SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Matrix, SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Array, SizedAbstractArray{Tuple{2,2,2,2}, Int}(ones(2,2,2,2))) == ones(2,2,2,2)
        # Conversion after reshaping
        @test Array(SizedAbstractMatrix{2,2,Int,1,Vector{Int}}([1,2,3,4])) == [1 3; 2 4]
    end

    @testset "promotion" begin
        @test @inferred(promote_type(SizedAbstractVector{1,Float64,1,Vector{Float64}}, SizedAbstractVector{1,BigFloat,1,Vector{BigFloat}})) == SizedAbstractVector{1,BigFloat,1,Vector{BigFloat}}
        @test @inferred(promote_type(SizedAbstractVector{2,Int,1,Vector{Int}}, SizedAbstractVector{2,Float64,1,Vector{Float64}})) === SizedAbstractVector{2,Float64,1,Vector{Float64}}
        @test @inferred(promote_type(SizedAbstractMatrix{2,3,Float32,2,Matrix{Float32}}, SizedAbstractMatrix{2,3,Complex{Float64},2,Matrix{Complex{Float64}}})) === SizedAbstractMatrix{2,3,Complex{Float64},2,Matrix{Complex{Float64}}}
    end

    @testset "dynamically sized axes" begin
        A = rand(Int, 2, 3, 4)
        B = SizedAbstractArray{Tuple{2,3,StaticArrays.Dynamic()}, Int, 3}(A)
        @test size(B) == size(A)
        @test axes(B) == (SOneTo(2), SOneTo(3), axes(A, 3))
        @test B[1,:,:] == A[1,:,:]
        @test_broken B[:,:,2] == A[:,:,2]
    end
end
