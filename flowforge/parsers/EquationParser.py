import sympy

class EquationParser:
    def __init__(self, equation: str, *coupled_variables):
        self._input_equation = equation
        self._expression = sympy.sympify(equation)

        potential_variable = ['x', 'y', 'z', 't'] + [i for i in coupled_variables]
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
        full_input = {'x': x, 'y': y, 'z': z, 't': t} \
            | {variable_name: value for variable_name, value in coupled_variables.items()}
        expression_input = self._generate_expression_input(full_input)
        expression = self._expression.subs(expression_input)

        return float(expression)
