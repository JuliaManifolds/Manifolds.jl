"""
    EmbeddedBackend <: AbstractDiffBackend

A type to specify / use differentiation by definiing the functions in the embedding

# Constructor
    EmbeddedBackend(; derivative=missing gradient=missing, jacobian=missing)
"""
struct EmbeddedBackend{D,G,J} <: AbstractDiffBackend
    gradient::G
    derivative::D
    jacobian::J
end

FiniteDiffBackend() = FiniteDiffBackend(Val(:central))

function _derivative(f, p, e::EmbeddedBackend) where {Method}
    return e.derivative(p)
end

function _derivative(f, p, ::EmbeddedBackend{Missing})
    throw(MissingException("The provided Embedded backend does not provide a derivative"))
end

function _gradient(f, p, e::EmbeddedBackend) where {Method}
    return e.gradient(p)
end

function _gradient(f, p, ::EmbeddedBackend{D,Missing}) where {D}
    throw(MissingException("The provided Embedded backend does not provide a gradient"))
end

function _jacobian(f, p, e::EmbeddedBackend)
    return e.jacobian(p)
end

function _jacobian(f, p, ::EmbeddedBackend{D,G,Missing}) where {D,G}
    throw(MissingException("The provided Embedded backend does not provide a jacobian"))
end
