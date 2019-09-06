using Manifolds
using Manifolds: HybridAbstractArray, HybridAbstractVector, HybridAbstractMatrix
using StaticArrays
using Test


struct ScalarTest end
Base.:(+)(x::Number, y::ScalarTest) = x
Broadcast.broadcastable(x::ScalarTest) = Ref(x)

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
        @test B[1,2,:] == @SVector [10, 11, 12, 13]
    end

    @testset "broadcasting" begin
        Ai = rand(Int, 2, 3, 4)
        Bi = HybridAbstractArray{Tuple{2,3,StaticArrays.Dynamic()}, Int, 3}(Ai)

        Af = rand(Float64, 2, 3, 4)
        Bf = HybridAbstractArray{Tuple{2,3,StaticArrays.Dynamic()}, Float64, 3}(Af)

        Bi[1,2,:] .= [110, 111, 112, 113]
        @test Bi[1,2,:] == @SVector [110, 111, 112, 113]

        @testset "Scalar Broadcast" begin
            @test Bf == @inferred(Bf .+ ScalarTest())
            @test Bf .+ 1 == @inferred(Bf .+ Ref(1))
        end

        @testset "AbstractArray-of-HybridAbstractArray with scalar math" begin
            v = [Bf]
            @test @inferred(v .* 1.0)::typeof(v) == v
        end
#=
        @testset "2x2 StaticMatrix with StaticVector" begin
            m = @SMatrix [1 2; 3 4]
            v = SVector(1, 4)
            @test @inferred(broadcast(+, m, v)) === @SMatrix [2 3; 7 8]
            @test @inferred(m .+ v) === @SMatrix [2 3; 7 8]
            @test @inferred(v .+ m) === @SMatrix [2 3; 7 8]
            @test @inferred(m .* v) === @SMatrix [1 2; 12 16]
            @test @inferred(v .* m) === @SMatrix [1 2; 12 16]
            @test @inferred(m ./ v) === @SMatrix [1 2; 3/4 1]
            @test @inferred(v ./ m) === @SMatrix [1 1/2; 4/3 1]
            @test @inferred(m .- v) === @SMatrix [0 1; -1 0]
            @test @inferred(v .- m) === @SMatrix [0 -1; 1 0]
            @test @inferred(m .^ v) === @SMatrix [1 2; 81 256]
            @test @inferred(v .^ m) === @SMatrix [1 1; 64 256]
            # Issue #546
            @test @inferred(m ./ (v .* v')) === @SMatrix [1.0 0.5; 0.75 0.25]
            testinf(m, v) = m ./ (v .* v')
            @test @inferred(testinf(m, v)) === @SMatrix [1.0 0.5; 0.75 0.25]
        end

        @testset "2x2 StaticMatrix with 1x2 StaticMatrix" begin
            # Issues #197, #242: broadcast between SArray and row-like SMatrix
            m1 = @SMatrix [1 2; 3 4]
            m2 = @SMatrix [1 4]
            @test @inferred(broadcast(+, m1, m2)) === @SMatrix [2 6; 4 8]
            @test @inferred(m1 .+ m2) === @SMatrix [2 6; 4 8]
            @test @inferred(m2 .+ m1) === @SMatrix [2 6; 4 8]
            @test @inferred(m1 .* m2) === @SMatrix [1 8; 3 16]
            @test @inferred(m2 .* m1) === @SMatrix [1 8; 3 16]
            @test @inferred(m1 ./ m2) === @SMatrix [1 1/2; 3 1]
            @test @inferred(m2 ./ m1) === @SMatrix [1 2; 1/3 1]
            @test @inferred(m1 .- m2) === @SMatrix [0 -2; 2 0]
            @test @inferred(m2 .- m1) === @SMatrix [0 2; -2 0]
            @test @inferred(m1 .^ m2) === @SMatrix [1 16; 3 256]
        end

        @testset "1x2 StaticMatrix with StaticVector" begin
            # Issues #197, #242: broadcast between SVector and row-like SMatrix
            m = @SMatrix [1 2]
            v = SVector(1, 4)
            @test @inferred(broadcast(+, m, v)) === @SMatrix [2 3; 5 6]
            @test @inferred(m .+ v) === @SMatrix [2 3; 5 6]
            @test @inferred(v .+ m) === @SMatrix [2 3; 5 6]
            @test @inferred(m .* v) === @SMatrix [1 2; 4 8]
            @test @inferred(v .* m) === @SMatrix [1 2; 4 8]
            @test @inferred(m ./ v) === @SMatrix [1 2; 1/4 1/2]
            @test @inferred(v ./ m) === @SMatrix [1 1/2; 4 2]
            @test @inferred(m .- v) === @SMatrix [0 1; -3 -2]
            @test @inferred(v .- m) === @SMatrix [0 -1; 3 2]
            @test @inferred(m .^ v) === @SMatrix [1 2; 1 16]
            @test @inferred(v .^ m) === @SMatrix [1 1; 4 16]
        end

                =#
        @testset "HybridAbstractMatrix with HybridAbstractMatrix" begin
            m1 = HybridAbstractMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
            m2 = HybridAbstractMatrix{2,StaticArrays.Dynamic()}([1 3; 4 5])
            @test @inferred(broadcast(+, m1, m2)) == @SMatrix [2 5; 7 9]
            @test @inferred(m1 .+ m2) == @SMatrix [2 5; 7 9]
            @test @inferred(m2 .+ m1) == @SMatrix [2 5; 7 9]
            @test @inferred(m1 .* m2) == @SMatrix [1 6; 12 20]
            @test @inferred(m2 .* m1) == @SMatrix [1 6; 12 20]
            # StaticArrays Issue #199: broadcast with empty SArray
            @test @inferred(HybridAbstractVector{1}([1]) .+ HybridAbstractVector{0,Int}([])) === SVector{0,Union{}}()
            @test_broken @inferred(HybridAbstractVector{0,Int}([]) .+ SVector(1)) === SVector{0,Union{}}()
            # StaticArrays Issue #200: broadcast with Adjoint
            @test @inferred(m1 .+ m2') == @SMatrix [2 6; 6 9]
            @test @inferred(m1 .+ transpose(m2)) == @SMatrix [2 6; 6 9]
            # StaticArrays Issue 382: infinite recursion in Base.Broadcast.broadcast_indices with Adjoint
            @test @inferred(HybridAbstractVector{2}([1,1])' .+ [1, 1]) == [2 2; 2 2]
            @test @inferred(transpose(HybridAbstractVector{2}([1,1])) .+ [1, 1]) == [2 2; 2 2]
            @test @inferred(HybridAbstractVector{StaticArrays.Dynamic()}([1,1])' .+ [1, 1]) == [2 2; 2 2]
            @test @inferred(transpose(HybridAbstractVector{StaticArrays.Dynamic()}([1,1])) .+ [1, 1]) == [2 2; 2 2]
        end

        @testset "HybridAbstractMatrix with Scalar" begin
            m = HybridAbstractMatrix{2,StaticArrays.Dynamic()}([1 2; 3 4])
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
            @test @inferred(1.0 .+ HybridAbstractMatrix{2,0,Float64}(zeros(2,0))) == HybridAbstractMatrix{2,0,Float64}(zeros(2,0))
            @test @inferred(1.0 .+ HybridAbstractMatrix{0,2,Float64}(zeros(0,2))) == HybridAbstractMatrix{0,2,Float64}(zeros(0,2))
            @test @inferred(1.0 .+ HybridAbstractArray{Tuple{2,StaticArrays.Dynamic(),0},Float64}(zeros(2,3,0))) == HybridAbstractArray{Tuple{2,StaticArrays.Dynamic(),0},Float64}(zeros(2,3,0))
            @test @inferred(HybridAbstractVector{0,Float64}(zeros(0)) .+ HybridAbstractMatrix{0,2,Float64}(zeros(0,2))) == HybridAbstractMatrix{0,2,Float64}(zeros(0,2))
            m = HybridAbstractMatrix{0,2,Float64}(zeros(0,2))
            @test @inferred(broadcast!(+, m, m, HybridAbstractVector{0,Float64}(zeros(0)))) == HybridAbstractMatrix{0,2,Float64}(zeros(0,2))
        end
#=
        @testset "Mutating broadcast!" begin
            # No setindex! error
            A = one(SMatrix{2, 2}); @test_throws ErrorException broadcast!(+, A, A, SVector(1, 4))
            A = one(MMatrix{2, 2}); @test @inferred(broadcast!(+, A, A, SVector(1, 4))) == @MMatrix [2 1; 4 5]
            A = one(MMatrix{2, 2}); @test @inferred(broadcast!(+, A, A, @SMatrix([1  4]))) == @MMatrix [2 4; 1 5]
            A = @MMatrix([1 0]); @test_throws DimensionMismatch broadcast!(+, A, A, SVector(1, 4))
            A = @MMatrix([1 0]); @test @inferred(broadcast!(+, A, A, @SMatrix([1 4]))) == @MMatrix [2 4]
            A = @MMatrix([1 0]); @test @inferred(broadcast!(+, A, A, 2)) == @MMatrix [3 2]
        end

        @testset "broadcast! with mixtures of SArray and Array" begin
            a = zeros(MVector{2}); @test @inferred(broadcast!(+, a, [1,2])) == [1,2]
            a = zeros(MMatrix{2,3}); @test @inferred(broadcast!(+, a, [1,2])) == [1 1 1; 2 2 2]
        end
=#
        @testset "eltype after broadcast" begin
            # test cases StaticArrays issue #198
            let a = HybridAbstractVector{4, Number}(Number[2, 2.0, 4//2, 2+0im])
                @test eltype(a .+ 2) == Number
                @test eltype(a .- 2) == Number
                @test eltype(a * 2) == Number
                @test eltype(a / 2) == Number
            end
            let a = HybridAbstractVector{3, Real}(Real[2, 2.0, 4//2])
                @test eltype(a .+ 2) == Real
                @test eltype(a .- 2) == Real
                @test eltype(a * 2) == Real
                @test eltype(a / 2) == Real
            end
            let a = HybridAbstractVector{3, Real}(Real[2, 2.0, 4//2])
                @test eltype(a .+ 2.0) == Float64
                @test eltype(a .- 2.0) == Float64
                @test eltype(a * 2.0) == Float64
                @test eltype(a / 2.0) == Float64
            end
            let a = broadcast(Float32, HybridAbstractVector{3}([3, 4, 5]))
                @test eltype(a) == Float32
            end
        end

        @testset "broadcast general scalars" begin
            # StaticArrays Issue #239 - broadcast with non-numeric element types
            @eval @enum Axis aX aY aZ
            @test (HybridAbstractVector{3}([aX,aY,aZ]) .== Ref(aX)) == HybridAbstractVector{3}([true,false,false])
            mv = HybridAbstractVector{3}([aX,aY,aZ])
            @test broadcast!(identity, mv, Ref(aX)) == HybridAbstractVector{3}([aX,aX,aX])
            @test mv == HybridAbstractVector{3}([aX,aX,aX])
        end

        @testset "broadcast! with Array destination" begin
            # Issue #385
            a = zeros(3, 3)
            b = HybridAbstractMatrix{3,StaticArrays.Dynamic()}([1 2 3; 4 5 6; 7 8 9])
            a .= b
            @test a == b

            c = HybridAbstractVector{3}([1, 2, 3])
            a .= c
            @test a == [1 1 1; 2 2 2; 3 3 3]

            d = HybridAbstractVector{4}([1, 2, 3, 4])
            @test_throws DimensionMismatch a .= d
        end
    end
end
