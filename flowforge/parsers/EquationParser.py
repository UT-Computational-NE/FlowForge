from typing import Optional
import re
import sympy


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

        potential_variable = ["x", "y", "z", "t"] + list(coupled_variables)
        variable_names_extracted_from_equation = [
            var for var in re.findall(r"[\w]+", equation) if any(char.isalpha() for char in var)
        ]
        self._variables = {
            var: sympy.symbols(var) for var in potential_variable if var in variable_names_extracted_from_equation
        }

    @property
    def inputEquation(self):
        return self._input_equation

    @inputEquation.setter
    def inputEquation(self, value: str):
        self._input_equation = value

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, value: sympy.Basic):
        self._expression = value

    @property
    def variables(self):
        return self._variables

    def performUnitConversion(
        self,
        scale_factor: Optional[float] = 1.0,
        shift_factor: Optional[float] = 0.0,
    ):
        """Apply unit conversion to the equation.

        This method modifies the equation by applying a linear transformation:
        new_equation = (scale_factor * original_equation) + shift_factor

        This transformation is useful for converting between different unit systems,
        such as from Celsius to Kelvin (using shift_factor) or from inches to
        meters (using scale_factor).

        Parameters
        ----------
        scale_factor : Optional[float]
            Multiplicative factor to apply to the equation, by default 1.
        shift_factor : Optional[float]
            Value to add to the equation after scaling, by default 0.

        Notes
        -----
        This method updates both the inputEquation string and the sympy expression
        object with the transformed equation. The original equation is replaced.

        Example
        -------
        To convert from Celsius to Kelvin:
        >>> parser = EquationParser("T")
        >>> parser.performUnitConversion(1, 273.15)
        This transforms T to (1 * (T)) + 273.15
        """
        scaled_equation = "(" + str(scale_factor) + " * (" + self.inputEquation + ")) + " + str(shift_factor)
        scaled_expression = sympy.sympify(scaled_equation)
        self.inputEquation = scaled_equation
        self.expression = scaled_expression

    def evaluate(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        t: Optional[float] = None,
        **coupled_variables,
    ):
        """Evaluate the equation with the provided variable values.

        This method substitutes variable values into the equation and evaluates
        the result. It intelligently handles variables by only substituting values
        for variables that are actually present in the equation.

        Parameters
        ----------
        x : Optional[float]
            Value for the 'x' spatial coordinate, by default None.
        y : Optional[float]
            Value for the 'y' spatial coordinate, by default None.
        z : Optional[float]
            Value for the 'z' spatial coordinate, by default None.
        t : Optional[float]
            Value for the time variable 't', by default None.
        **coupled_variables : dict
            Additional variable values for any coupled variables provided during
            initialization. Keys should match the variable names.

        Returns
        -------
        float
            The evaluated result of the equation with the provided variable values.

        Notes
        -----
        Variables not used in the equation are ignored, so you can safely provide
        values for all possible variables without causing errors.

        Examples
        --------
        >>> parser = EquationParser("x**2 + y", "temperature")
        >>> parser.evaluate(x=2, y=3, z=4, temperature=100)
        7.0

        In this example, z=4 is ignored because 'z' isn't in the equation.
        """
        full_input = {"x": x, "y": y, "z": z, "t": t} | dict(coupled_variables)
        reduced_input = {self._variables[var]: full_input[var] for var in [*self._variables.keys()]}
        expression = self._expression.subs(reduced_input)

        return float(expression)
