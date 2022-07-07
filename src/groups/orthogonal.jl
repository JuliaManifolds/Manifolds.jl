@doc raw"""
    Orthogonal{n} <: GroupManifold{ℝ,OrthogonalMatrices{n},MultiplicationOperation}

Orthogonal group $\mathrm{O}(n)$ represented by orthogonal matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const Orthogonal{n} = AbstractUnitaryMultiplicationGroup{n,ℝ,OrthogonalMatrices{n}}

Orthogonal(n) = Orthogonal{n}()

@doc raw"""
     get_coordinates(G::Orthogonal, p, X)
 Extract the unique tangent vector components $X^i$ at point `p` on [`Orthogonal`](@ref)
 $\mathrm{O}(n)$ from the matrix representation `X` of the tangent vector.
 The basis on the Lie algebra $𝔬(n)$ is chosen such that
 for $\mathrm{O}(2)$, $X^1 = θ = X_{21}$ is the angle of rotation, and
 for $\mathrm{O}(3)$, $(X^1, X^2, X^3) = (X_{32}, X_{13}, X_{21}) = θ u$ is the
 angular velocity and axis-angle representation, where $u$ is the unit vector
 along the axis of rotation.
 For $\mathrm{O}(n)$ where $n ≥ 4$, the additional elements of $X^i$ are
 $X^{j (j - 3)/2 + k + 1} = X_{jk}$, for $j ∈ [4,n], k ∈ [1,j)$.
 """
 get_coordinates(::Orthogonal, ::Any...)
 get_coordinates(::Orthogonal{2}, p, X, ::DefaultOrthogonalBasis) = [X[2]]

 function get_coordinates!(::Orthogonal{2}, Xⁱ, p, X, ::DefaultOrthogonalBasis)
     Xⁱ[1] = X[2]
     return Xⁱ
 end
 function get_coordinates!(G::Orthogonal{n}, Xⁱ, p, X, B::DefaultOrthogonalBasis) where {n}
     @inbounds begin
         Xⁱ[1] = X[3, 2]
         Xⁱ[2] = X[1, 3]
         Xⁱ[3] = X[2, 1]

         k = 4
         for i in 4:n, j in 1:(i - 1)
             Xⁱ[k] = X[i, j]
             k += 1
         end
     end
     return Xⁱ
 end
 function get_coordinates!(G::Orthogonal, Xⁱ, p, X, ::DefaultOrthonormalBasis)
     get_coordinates!(G, Xⁱ, p, X, DefaultOrthogonalBasis())
     T = eltype(Xⁱ)
     Xⁱ .*= sqrt(T(2))
     return Xⁱ
 end

 @doc raw"""
     get_vector(G::Orthogonal, p, Xⁱ, B:: DefaultOrthogonalBasis)
 Convert the unique tangent vector components `Xⁱ` at point `p` on [`Orthogonal`](@ref)
 group $\mathrm{O}(n)$ to the matrix representation $X$ of the tangent vector. See
 [`get_coordinates`](@ref get_coordinates(::Orthogonal, ::Any...)) for the conventions used.
 """
 get_vector(::Orthogonal, ::Any...)

 function get_vector!(G::Orthogonal{2}, X, p, Xⁱ, B::DefaultOrthogonalBasis)
     return get_vector!(G, X, p, Xⁱ[1], B)
 end
 function get_vector!(::Orthogonal{2}, X, p, Xⁱ::Real, ::DefaultOrthogonalBasis)
     @assert size(X) == (2, 2)
     @inbounds begin
         X[1, 1] = X[2, 2] = 0
         X[2, 1] = Xⁱ
         X[1, 2] = -Xⁱ
     end
     return X
 end
 function get_vector!(G::Orthogonal{n}, X, p, Xⁱ, ::DefaultOrthogonalBasis) where {n}
     @assert size(X) == (n, n)
     @assert length(Xⁱ) == manifold_dimension(G)
     @inbounds begin
         X[1, 1] = 0
         X[1, 2] = -Xⁱ[3]
         X[1, 3] = Xⁱ[2]
         X[2, 1] = Xⁱ[3]
         X[2, 2] = 0
         X[2, 3] = -Xⁱ[1]
         X[3, 1] = -Xⁱ[2]
         X[3, 2] = Xⁱ[1]
         X[3, 3] = 0
         k = 4
         for i in 4:n
             for j in 1:(i - 1)
                 X[i, j] = Xⁱ[k]
                 X[j, i] = -Xⁱ[k]
                 k += 1
             end
             X[i, i] = 0
         end
     end
     return X
 end
 function get_vector!(G::Orthogonal, X, p, Xⁱ, B::DefaultOrthonormalBasis)
     get_vector!(G, X, p, Xⁱ, DefaultOrthogonalBasis())
     T = eltype(X)
     X .*= inv(sqrt(T(2)))
     return X
 end