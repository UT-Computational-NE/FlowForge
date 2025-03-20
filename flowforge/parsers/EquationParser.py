import re
import sympy

from typing import Dict, List, Optional, Union, Any, Tuple, Callable

class EquationParser:
    """
    Class for handling the parsing of input equation.

    This object takes in a string-equation and deciphers the variables
    used and generates an expression using sympy.

    The class automatically creates 4 independent variables for use:
      1. 'x': x-spatial coordinate
      2. 'y': y-spatial coordinate
      3. 'z': z-spatial coordinate
      4. 't': time
    The class also allows for coupled variables to declared. They should
    be input as a list of variable names (i.e. ['temperature', 'pressure'])

    The 'generate' method takes in keyword arguments for the variables,
    disregarding variables not used in the original expression (those of
    which the equation is not dependent on).

    Parameters
    ----------
      -> equation : string
      -> coupled_variables : *args (list)

    Attributes
    ----------
      -> inputEquation : string
      -> expression : sympify-object
      -> variables : dict{string : sympy.symbol}

    Methods
    -------
      -> _extract_variable_name : (private) extracts all variables from the
            input equation
      -> _generate_expression_input : (private) extracts valid variables from
            the input arguments in 'evaluate' and ignores unused variables.
      -> performUnitConversion : adds a scaling factor and shifting factor to
            the expression in the form
                        'new_eqn = (scaler * original+eqn) + shifter'.
            The new equation is them turned into an expression that may be
            properly evaluated.
      -> evaluate : takes in variable values and outputs the solution to the
            evaluated function.
    """
    def __init__(self, equation: str, *coupled_variables):
        self._input_equation = equation
        self._expression = sympy.sympify(equation)

        potential_variable = ['x', 'y', 'z', 't'] + list(coupled_variables)
        variable_names_extracted_from_equation = [var for var in re.findall(r'[\w]+', equation)
                                                  if any(char.isalpha() for char in var)]
        self._variables = {var: sympy.symbols(var) for var in potential_variable
                           if var in variable_names_extracted_from_equation}

    @property


    def inputEquation(self) -> Any:
        return self._input_equation

    @inputEquation.setter
    def inputEquation(self, value):
        self._input_equation = value

    @property


    def expression(self) -> Any:
        return self._expression

    @expression.setter
    def expression(self, value):
        self._expression = value

    @property


    def variables(self) -> Any:
        return self._variables

    def performUnitConversion(self, scale_factor=1, shift_factor=0,):
        scaled_equation = "(" + str(scale_factor) + " * (" + self.inputEquation + ")) + " + str(shift_factor)
        scaled_expression = sympy.sympify(scaled_equation)
        self.inputEquation = scaled_equation
        self.expression = scaled_expression

    def evaluate(self, x=None, y=None, z=None, t=None, **coupled_variables):
        full_input = {'x': x, 'y': y, 'z': z, 't': t} | dict(coupled_variables)
        reduced_input = {self._variables[var]: full_input[var] for var in [*self._variables.keys()]}
        expression = self._expression.subs(reduced_input)

        return float(expression)
