import abc
import sympy
import numpy as np

class EquationParser:
    def __init__(self, equation: str, *coupled_variables):
        self._input_equation = equation
        self._expression = sympy.sympify(equation)
        self._t = sympy.symbols('t')
        self._x = sympy.symbols('x')
        self._y = sympy.symbols('y')
        self._z = sympy.symbols('z')
        self._coupled_variables = {i: sympy.symbols(i) for i in coupled_variables}

    @property
    def inputEquation(self):
        return self._input_equation

    @property
    def expression(self):
        return self._expression

    @property
    def time(self):
        return self._t

    @property
    def xCoord(self):
        return self._x

    @property
    def yCoord(self):
        return self._y

    @property
    def zCoord(self):
        return self._z

    @property
    def coupledVariables(self):
        return self._coupled_variables

    def evaluate(self, x=None, y=None, z=None, t=None, **kwargs):
        expression = self.expression
        if x is not None:
            expression = expression.subs(self.xCoord, x)
        if y is not None:
            expression = expression.subs(self.yCoord, y)
        if z is not None:
            expression = expression.subs(self.zCoord, z)
        if t is not None:
            expression = expression.subs(self.time, t)

        for variable_name, value in kwargs.items():
            expression = expression.subs(self.coupledVariables[variable_name], value)

        return float(expression)
