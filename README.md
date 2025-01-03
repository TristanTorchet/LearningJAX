# LearningJAX

`eqx1_Linear.ipynb`:
- eqx.Linear with autograd
- writing a custom linear layer with jax (abc.ABC, abc.abstractmethod, @dataclasses.dataclass) (all eqx.modules are written in this way)
- `py1_decorators.ipynb`: essentially a decorator is a wrapper function which adds (modifies) functionality to the original function
- `py2_dataclasses.ipynb`:
  - abstract classes are meant to be inherited by other classes
  - abstract methods are meant to enforce the overwritting by the inheriting classes 
  - the dataclass decorator is a shortcut to create classes with __init__, __repr__, __eq__, __hash__ methods