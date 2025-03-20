import re
from typing import Any
import sympy


class EquationParser:
    """Parser for mathematical equations with symbolic variables.

    This class handles the parsing of string-based mathematical equations,
    identifying variables and generating evaluable symbolic expressions using sympy.

    The class automatically recognizes four standard independent variables:
    'x', 'y', and 'z' for spatial coordinates, and 't' for time. Additional
    coupled variables can be declared when initializing the parser.

    When evaluating expressions, the class intelligently disregards variables
    not used in the original equation.

    Parameters
    ----------
    equation : str
        The mathematical equation as a string.
    *coupled_variables : str
        Optional additional variable names beyond the default x, y, z, t.

    Attributes
    ----------
    inputEquation : str
        The current equation string, accessible via inputEquation property.
    expression : sympy.Expr
        The sympy expression object for the equation, accessible via expression property.
    variables : dict
        Dictionary mapping variable names to sympy symbols, accessible via variables property.
        Format: {str: sympy.Symbol}.

    Notes
    -----
    The class automatically extracts variables from the equation and only includes
    variables that are actually present in the equation. This means if you provide
    'x', 'y', 'z', 't' plus other coupled variables, but only some are used in the
    equation, only those variables will be included in the internal variables dictionary.
    """

    def __init__(self, equation: str, *coupled_variables):
        """Initialize the EquationParser with an equation and optional variables.

        Parameters
        ----------
        equation : str
            The mathematical equation as a string.
        *coupled_variables : str
            Optional additional variable names beyond the default x, y, z, t.

        Notes
        -----
        The initialization process:
        1. Stores the input equation as a string
        2. Converts the equation to a sympy expression
        3. Identifies all potential variables (x, y, z, t plus any coupled vars)
        4. Extracts actual variable names from the equation
        5. Creates symbols only for variables that are actually used in the equation
        """
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
    def inputEquation(self) -> Any:
        """str: Get the current equation string.

        Returns the string representation of the current equation.
        """
        return self._input_equation

    @inputEquation.setter
    def inputEquation(self, value):
        """Set the current equation string.

        Parameters
        ----------
        value : str
            The new equation string to set.
        """
        self._input_equation = value

    @property
    def expression(self) -> Any:
        """sympy.Expr: Get the current sympy expression object.

        Returns the sympy expression object for the current equation.
        """
        return self._expression

    @expression.setter
    def expression(self, value):
        """Set the current sympy expression.

        Parameters
        ----------
        value : sympy.Expr
            The new sympy expression to set.
        """
        self._expression = value

    @property
    def variables(self) -> Any:
        """dict: Get the dictionary of variables in the equation.

        Returns a dictionary mapping variable names (str) to sympy symbols.
        Only includes variables that are actually used in the equation.
        """
        return self._variables

    def performUnitConversion(
        self,
        scale_factor=1,
        shift_factor=0,
    ):
        """Apply unit conversion to the equation.

        This method modifies the equation by applying a linear transformation:
        new_equation = (scale_factor * original_equation) + shift_factor

        This transformation is useful for converting between different unit systems,
        such as from Celsius to Kelvin (using shift_factor) or from inches to
        meters (using scale_factor).

        Parameters
        ----------
        scale_factor : float, optional
            Multiplicative factor to apply to the equation, by default 1.
        shift_factor : float, optional
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

    def evaluate(self, x=None, y=None, z=None, t=None, **coupled_variables):
        """Evaluate the equation with the provided variable values.

        This method substitutes variable values into the equation and evaluates
        the result. It intelligently handles variables by only substituting values
        for variables that are actually present in the equation.

        Parameters
        ----------
        x : float, optional
            Value for the 'x' spatial coordinate, by default None.
        y : float, optional
            Value for the 'y' spatial coordinate, by default None.
        z : float, optional
            Value for the 'z' spatial coordinate, by default None.
        t : float, optional
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
