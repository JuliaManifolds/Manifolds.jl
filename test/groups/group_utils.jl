struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: AbstractManifold{ℝ} end

struct NotImplementedGroupDecorator{M} <: AbstractDecoratorManifold{ℝ,GroupDecoratorType}
    manifold::M
end
