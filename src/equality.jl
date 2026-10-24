#
# Comparison by value of types that store mutable data, like arrays, or wrap other objects
# that may do so, see `ManifoldsBase.@fieldwise_equality`. Types whose comparison may be
# performance relevant list their fields, which yields type stable methods. Type variables
# that stand for types are given with their upper bounds, number systems 𝔽 and the order D of
# a tensor are values and have none.
#

# manifolds and methods
ManifoldsBase.@fieldwise_equality FiberBundle (type, manifold, vector_transport)
ManifoldsBase.@fieldwise_equality FiberBundleProductVectorTransport
ManifoldsBase.@fieldwise_equality ParametricSurface

# atlases, their charts and bases
ManifoldsBase.@fieldwise_equality InducedBasis{𝔽} 𝔽 (vs, A, i)
ManifoldsBase.@fieldwise_equality ParametricSurfaceAtlas
ManifoldsBase.@fieldwise_equality PowerAtlas
ManifoldsBase.@fieldwise_equality ProductAtlas
ManifoldsBase.@fieldwise_equality RetractionAtlas{𝔽} 𝔽 (retr, invretr, basis)
ManifoldsBase.@fieldwise_equality StiefelChart

# basis data
ManifoldsBase.@fieldwise_equality FiberBundleBasisData
ManifoldsBase.@fieldwise_equality HOSVDBasis

# points, tangent and cotangent vectors
ManifoldsBase.@fieldwise_equality HOSVD{T, D} (T <: Number, D) (U, core, σ)
ManifoldsBase.@fieldwise_equality RieszRepresenterCotangentVector
ManifoldsBase.@fieldwise_equality SPDPoint (eigen,)
ManifoldsBase.@fieldwise_equality StiefelFactorization
ManifoldsBase.@fieldwise_equality SVDMPoint (U, S, Vt)
ManifoldsBase.@fieldwise_equality TuckerPoint{T, D} (T <: Number, D) (hosvd,)
ManifoldsBase.@fieldwise_equality TuckerTangentVector (Ċ, U̇)
ManifoldsBase.@fieldwise_equality UMVTangentVector (U, M, Vt)
ManifoldsBase.@fieldwise_equality VeronesePoint (radial, spherical)
ManifoldsBase.@fieldwise_equality VeroneseTangentVector (radial, spherical)
