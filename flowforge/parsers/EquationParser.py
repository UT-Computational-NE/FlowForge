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

        potential_variable = ['x', 'y', 'z', 't'] + list(coupled_variables)
        variable_names = self._extract_variable_name(equation)
        self._variables = {var: sympy.symbols(var) for var in potential_variable
                           if var in variable_names}

    @property
    def inputEquation(self):
        return self._input_equation

    @inputEquation.setter
    def inputEquation(self, value):
        self._input_equation = value

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, value):
        self._expression = value

    @property
    def variables(self):
        return self._variables

    def _extract_variable_name(self, equation):
        variables = ''.join(character if character.isalpha()
                            else ' ' for character in equation).split()
        return variables

    def _generate_expression_input(self, all_input: dict):
        expression_input = {self._variables[var]: all_input[var]
                            for var in [*self._variables.keys()]}
        return expression_input

    def performUnitConversion(self, scale_factor=1, shift_factor=0,):
        scaled_equation = "(" + str(scale_factor) + " * (" + self.inputEquation + ")) + " + str(shift_factor)
        scaled_expression = sympy.sympify(scaled_equation)
        self.inputEquation = scaled_equation
        self.expression = scaled_expression

    def evaluate(self, x=None, y=None, z=None, t=None, **coupled_variables):
        full_input = {'x': x, 'y': y, 'z': z, 't': t} | dict(coupled_variables)
        expression_input = self._generate_expression_input(full_input)
        expression = self._expression.subs(expression_input)

        return float(expression)
