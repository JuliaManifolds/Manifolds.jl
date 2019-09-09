using Manifolds
using Manifolds: HybridArray, HybridVector, HybridMatrix
using StaticArrays
using Test


struct ScalarTest end
Base.:(+)(x::Number, y::ScalarTest) = x
Broadcast.broadcastable(x::ScalarTest) = Ref(x)

@testset "HybridArray" begin
    @testset "Inner Constructors" begin
        @test HybridArray{Tuple{2}, Int, 1, 1, Vector{Int}}((3, 4)).data == [3, 4]
        @test HybridArray{Tuple{2}, Int, 1}([3, 4]).data == [3, 4]
        @test HybridArray{Tuple{2, 2}, Int, 2}(collect(3:6)).data == collect(3:6)
        @test size(HybridArray{Tuple{4, 5}, Int, 2}(undef).data) == (4, 5)
        @test size(HybridArray{Tuple{4, 5}, Int}(undef).data) == (4, 5)

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1}([2 3])

        # Bad parameters
        @test_throws Exception HybridArray{Tuple{1},Int,2}(undef)
        @test_throws Exception SArray{Tuple{3, 4},Int,1}(undef)

        # Parameter/input size mismatch
        @test_throws Exception HybridArray{Tuple{1},Int,2}([2; 3])
        @test_throws Exception HybridArray{Tuple{1},Int,2}((2, 3))
    end

    @testset "Outer Constructors" begin
        # From Array
        @test @inferred(HybridArray{Tuple{2},Float64,1}([1,2]))::HybridArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(HybridArray{Tuple{2},Float64}([1,2]))::HybridArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(HybridArray{Tuple{2}}([1,2]))::HybridArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(HybridArray{Tuple{2,2}}([1 2;3 4]))::HybridArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]

        # Uninitialized
        @test @inferred(HybridArray{Tuple{2,2},Int,2}(undef)) isa HybridArray{Tuple{2,2},Int,2,2}
        @test @inferred(HybridArray{Tuple{2,2},Int}(undef)) isa HybridArray{Tuple{2,2},Int,2,2}

        # From Tuple
        @test @inferred(HybridArray{Tuple{2},Float64,1,1,Vector{Float64}}((1,2)))::HybridArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(HybridArray{Tuple{2},Float64}((1,2)))::HybridArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(HybridArray{Tuple{2}}((1,2)))::HybridArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(HybridArray{Tuple{2,2}}((1,2,3,4)))::HybridArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
    end

    @testset "HybridVector and HybridMatrix" begin
        @test @inferred(HybridVector{2}([1,2]))::HybridArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(HybridVector{2}((1,2)))::HybridArray{Tuple{2},Int,1,1} == [1,2]
        # Reshaping
        @test @inferred(HybridVector{2}([1 2]))::HybridArray{Tuple{2},Int,1,2} == [1,2]
        # Back to Vector
        @test Vector(HybridVector{2}((1,2))) == [1,2]
        @test convert(Vector, HybridVector{2}((1,2))) == [1,2]

        @test @inferred(HybridMatrix{2,2}([1 2; 3 4]))::HybridArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
        # Reshaping
        @test @inferred(HybridMatrix{2,2}((1,2,3,4)))::HybridArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
        # Back to Matrix
        @test Matrix(HybridMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
        @test convert(Matrix, HybridMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
    end

    # setindex
    sa = HybridArray{Tuple{2}, Int, 1}([3, 4])
    sa[1] = 2
    @test sa.data == [2, 4]

    @testset "aliasing" begin
        a1 = rand(4)
        a2 = copy(a1)
        sa1 = HybridVector{4}(a1)
        sa2 = HybridVector{4}(a2)
        @test Base.mightalias(a1, sa1)
        @test Base.mightalias(sa1, HybridVector{4}(a1))
        @test !Base.mightalias(a2, sa1)
        @test !Base.mightalias(sa1, HybridVector{4}(a2))
        @test Base.mightalias(sa1, view(sa1, 1:2))
        @test Base.mightalias(a1, view(sa1, 1:2))
        @test Base.mightalias(sa1, view(a1, 1:2))
    end

    @testset "back to Array" begin
        @test Array(HybridArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int}(HybridArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int, 1}(HybridArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Vector(HybridArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test convert(Vector, HybridArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test Matrix(SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Matrix, SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Array, HybridArray{Tuple{2,2,2,2}, Int}(ones(2,2,2,2))) == ones(2,2,2,2)
        # Conversion after reshaping
        @test Array(HybridMatrix{2,2,Int,1,Vector{Int}}([1,2,3,4])) == [1 3; 2 4]
    end

    @testset "promotion" begin
        @test @inferred(promote_type(HybridVector{1,Float64,1,Vector{Float64}}, HybridVector{1,BigFloat,1,Vector{BigFloat}})) == HybridVector{1,BigFloat,1,Vector{BigFloat}}
        @test @inferred(promote_type(HybridVector{2,Int,1,Vector{Int}}, HybridVector{2,Float64,1,Vector{Float64}})) === HybridVector{2,Float64,1,Vector{Float64}}
        @test @inferred(promote_type(HybridMatrix{2,3,Float32,2,Matrix{Float32}}, HybridMatrix{2,3,Complex{Float64},2,Matrix{Complex{Float64}}})) === HybridMatrix{2,3,Complex{Float64},2,Matrix{Complex{Float64}}}
    end

    @testset "dynamically sized axes" begin
        A = rand(Int, 2, 3, 4)
        B = HybridArray{Tuple{2,3,StaticArrays.Dynamic()}, Int, 3}(A)
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
        @test B[1,2,:] == @SVector [10, 11, 12, 13]
    end

    @testset "broadcasting" begin
        Ai = rand(Int, 2, 3, 4)
        Bi = HybridArray{Tuple{2,3,StaticArrays.Dynamic()}, Int, 3}(Ai)

        Af = rand(Float64, 2, 3, 4)
        Bf = HybridArray{Tuple{2,3,StaticArrays.Dynamic()}, Float64, 3}(Af)

        Bi[1,2,:] .= [110, 111, 112, 113]
        @test Bi[1,2,:] == @SVector [110, 111, 112, 113]

        @testset "Scalar Broadcast" begin
            @test Bf == @inferred(Bf .+ ScalarTest())
            @test Bf .+ 1 == @inferred(Bf .+ Ref(1))
        end

        @testset "AbstractArray-of-HybridArray with scalar math" begin
            v = [Bf]
            @test @inferred(v .* 1.0)::typeof(v) == v
        end

        @testset "2x2 HybridMatrix with HybridVector" begin
            m = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
            v = HybridVector{2}([1, 4])
            @test @inferred(broadcast(+, m, v)) == @SMatrix [2 3; 7 8]
            @test @inferred(m .+ v) == @SMatrix [2 3; 7 8]
            @test @inferred(v .+ m) == @SMatrix [2 3; 7 8]
            @test @inferred(m .* v) == @SMatrix [1 2; 12 16]
            @test @inferred(v .* m) == @SMatrix [1 2; 12 16]
            @test @inferred(m ./ v) == @SMatrix [1 2; 3/4 1]
            @test @inferred(v ./ m) == @SMatrix [1 1/2; 4/3 1]
            @test @inferred(m .- v) == @SMatrix [0 1; -1 0]
            @test @inferred(v .- m) == @SMatrix [0 -1; 1 0]
            @test @inferred(m .^ v) == @SMatrix [1 2; 81 256]
            @test @inferred(v .^ m) == @SMatrix [1 1; 64 256]
            # StaticArrays Issue #546
            @test @inferred(m ./ (v .* v')) == @SMatrix [1.0 0.5; 0.75 0.25]
            testinf(m, v) = m ./ (v .* v')
            @test @inferred(testinf(m, v)) == @SMatrix [1.0 0.5; 0.75 0.25]
        end

        @testset "2x2 HybridMatrix with 1x2 HybridMatrix" begin
            # StaticArrays Issues #197, #242: broadcast between SArray and row-like SMatrix
            m1 = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
            m2 = HybridMatrix{1,StaticArrays.Dynamic()}([1 4])
            @test @inferred(broadcast(+, m1, m2)) == @SMatrix [2 6; 4 8]
            @test @inferred(m1 .+ m2) == @SMatrix [2 6; 4 8]
            @test @inferred(m2 .+ m1) == @SMatrix [2 6; 4 8]
            @test @inferred(m1 .* m2) == @SMatrix [1 8; 3 16]
            @test @inferred(m2 .* m1) == @SMatrix [1 8; 3 16]
            @test @inferred(m1 ./ m2) == @SMatrix [1 1/2; 3 1]
            @test @inferred(m2 ./ m1) == @SMatrix [1 2; 1/3 1]
            @test @inferred(m1 .- m2) == @SMatrix [0 -2; 2 0]
            @test @inferred(m2 .- m1) == @SMatrix [0 2; -2 0]
            @test @inferred(m1 .^ m2) == @SMatrix [1 16; 3 256]
        end

        @testset "1x2 HybridMatrix with SVector" begin
            # StaticArrays Issues #197, #242: broadcast between SVector and row-like SVector
            m = HybridMatrix{1,StaticArrays.Dynamic()}([1 2])
            v = SVector(1, 4)
            @test @inferred(broadcast(+, m, v)) == @SMatrix [2 3; 5 6]
            @test @inferred(m .+ v) == @SMatrix [2 3; 5 6]
            @test @inferred(v .+ m) == @SMatrix [2 3; 5 6]
            @test @inferred(m .* v) == @SMatrix [1 2; 4 8]
            @test @inferred(v .* m) == @SMatrix [1 2; 4 8]
            @test @inferred(m ./ v) == @SMatrix [1 2; 1/4 1/2]
            @test @inferred(v ./ m) == @SMatrix [1 1/2; 4 2]
            @test @inferred(m .- v) == @SMatrix [0 1; -3 -2]
            @test @inferred(v .- m) == @SMatrix [0 -1; 3 2]
            @test @inferred(m .^ v) == @SMatrix [1 2; 1 16]
            @test @inferred(v .^ m) == @SMatrix [1 1; 4 16]
        end

        @testset "HybridMatrix with HybridMatrix" begin
            m1 = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
            m2 = HybridMatrix{2,StaticArrays.Dynamic()}([1 3; 4 5])
            @test @inferred(broadcast(+, m1, m2)) == @SMatrix [2 5; 7 9]
            @test @inferred(m1 .+ m2) == @SMatrix [2 5; 7 9]
            @test @inferred(m2 .+ m1) == @SMatrix [2 5; 7 9]
            @test @inferred(m1 .* m2) == @SMatrix [1 6; 12 20]
            @test @inferred(m2 .* m1) == @SMatrix [1 6; 12 20]
            # StaticArrays Issue #199: broadcast with empty SArray
            @test @inferred(HybridVector{1}([1]) .+ HybridVector{0,Int}([])) === SVector{0,Union{}}()
            @test_broken @inferred(HybridVector{0,Int}([]) .+ SVector(1)) === SVector{0,Union{}}()
            # StaticArrays Issue #200: broadcast with Adjoint
            @test @inferred(m1 .+ m2') == @SMatrix [2 6; 6 9]
            @test @inferred(m1 .+ transpose(m2)) == @SMatrix [2 6; 6 9]
            # StaticArrays Issue 382: infinite recursion in Base.Broadcast.broadcast_indices with Adjoint
            @test @inferred(HybridVector{2}([1,1])' .+ [1, 1]) == [2 2; 2 2]
            @test @inferred(transpose(HybridVector{2}([1,1])) .+ [1, 1]) == [2 2; 2 2]
            @test @inferred(HybridVector{StaticArrays.Dynamic()}([1,1])' .+ [1, 1]) == [2 2; 2 2]
            @test @inferred(transpose(HybridVector{StaticArrays.Dynamic()}([1,1])) .+ [1, 1]) == [2 2; 2 2]
        end

        @testset "HybridMatrix with Scalar" begin
            m = HybridMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
            @test @inferred(broadcast(+, m, 2)) == @SMatrix [3 4; 5 6]
            @test @inferred(m .+ 2) == @SMatrix [3 4; 5 6]
            @test @inferred(2 .+ m) == @SMatrix [3 4; 5 6]
            @test @inferred(m .* 2) == @SMatrix [2 4; 6 8]
            @test @inferred(2 .* m) == @SMatrix [2 4; 6 8]
            @test @inferred(m ./ 2) == @SMatrix [1/2 1; 3/2 2]
            @test @inferred(2 ./ m) == @SMatrix [2 1; 2/3 1/2]
            @test @inferred(m .- 2) == @SMatrix [-1 0; 1 2]
            @test @inferred(2 .- m) == @SMatrix [1 0; -1 -2]
            @test @inferred(m .^ 2) == @SMatrix [1 4; 9 16]
            @test @inferred(2 .^ m) == @SMatrix [2 4; 8 16]
        end
        @testset "Empty arrays" begin
            @test @inferred(1.0 .+ HybridMatrix{2,0,Float64}(zeros(2,0))) == HybridMatrix{2,0,Float64}(zeros(2,0))
            @test @inferred(1.0 .+ HybridMatrix{0,2,Float64}(zeros(0,2))) == HybridMatrix{0,2,Float64}(zeros(0,2))
            @test @inferred(1.0 .+ HybridArray{Tuple{2,StaticArrays.Dynamic(),0},Float64}(zeros(2,3,0))) == HybridArray{Tuple{2,StaticArrays.Dynamic(),0},Float64}(zeros(2,3,0))
            @test @inferred(HybridVector{0,Float64}(zeros(0)) .+ HybridMatrix{0,2,Float64}(zeros(0,2))) == HybridMatrix{0,2,Float64}(zeros(0,2))
            m = HybridMatrix{0,2,Float64}(zeros(0,2))
            @test @inferred(broadcast!(+, m, m, HybridVector{0,Float64}(zeros(0)))) == HybridMatrix{0,2,Float64}(zeros(0,2))
        end

        @testset "Mutating broadcast!" begin
            # No setindex! error
            A = HybridMatrix{2,StaticArrays.Dynamic()}([1 0; 0 1])
            @test @inferred(broadcast!(+, A, A, SVector(1, 4))) == @MMatrix [2 1; 4 5]
            A = HybridMatrix{2,StaticArrays.Dynamic()}([1 0; 0 1])
            @test @inferred(broadcast!(+, A, A, @SMatrix([1  4]))) == @MMatrix [2 4; 1 5]
            A = HybridMatrix{1,StaticArrays.Dynamic()}([1 0])
            @test_throws DimensionMismatch broadcast!(+, A, A, SVector(1, 4))
            A = HybridMatrix{1,StaticArrays.Dynamic()}([1 0])
            @test @inferred(broadcast!(+, A, A, @SMatrix([1 4]))) == @MMatrix [2 4]
            A = HybridMatrix{1,StaticArrays.Dynamic()}([1 0])
            @test @inferred(broadcast!(+, A, A, 2)) == @MMatrix [3 2]
        end

        @testset "broadcast! with mixtures of SArray and Array" begin
            A = HybridVector{StaticArrays.Dynamic()}([0, 0])
            @test @inferred(broadcast!(+, A, [1,2])) == [1,2]
        end

        @testset "eltype after broadcast" begin
            # test cases StaticArrays issue #198
            let a = HybridVector{4, Number}(Number[2, 2.0, 4//2, 2+0im])
                @test eltype(a .+ 2) == Number
                @test eltype(a .- 2) == Number
                @test eltype(a * 2) == Number
                @test eltype(a / 2) == Number
            end
            let a = HybridVector{3, Real}(Real[2, 2.0, 4//2])
                @test eltype(a .+ 2) == Real
                @test eltype(a .- 2) == Real
                @test eltype(a * 2) == Real
                @test eltype(a / 2) == Real
            end
            let a = HybridVector{3, Real}(Real[2, 2.0, 4//2])
                @test eltype(a .+ 2.0) == Float64
                @test eltype(a .- 2.0) == Float64
                @test eltype(a * 2.0) == Float64
                @test eltype(a / 2.0) == Float64
            end
            let a = broadcast(Float32, HybridVector{3}([3, 4, 5]))
                @test eltype(a) == Float32
            end
        end

        @testset "broadcast general scalars" begin
            # StaticArrays Issue #239 - broadcast with non-numeric element types
            @eval @enum Axis aX aY aZ
            @test (HybridVector{3}([aX,aY,aZ]) .== Ref(aX)) == HybridVector{3}([true,false,false])
            mv = HybridVector{3}([aX,aY,aZ])
            @test broadcast!(identity, mv, Ref(aX)) == HybridVector{3}([aX,aX,aX])
            @test mv == HybridVector{3}([aX,aX,aX])
        end

        @testset "broadcast! with Array destination" begin
            # Issue #385
            a = zeros(3, 3)
            b = HybridMatrix{3,StaticArrays.Dynamic()}([1 2 3; 4 5 6; 7 8 9])
            a .= b
            @test a == b

            c = HybridVector{3}([1, 2, 3])
            a .= c
            @test a == [1 1 1; 2 2 2; 3 3 3]

            d = HybridVector{4}([1, 2, 3, 4])
            @test_throws DimensionMismatch a .= d
        end
    end
end
