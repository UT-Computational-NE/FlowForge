import sympy

class EquationParser:
    def __init__(self, equation: str, *coupled_variables):
        self._input_equation = equation
        self._expression = sympy.sympify(equation)

        potential_variable = ['x', 'y', 'z', 't'] + [i for i in coupled_variables]
        variable_names = self._extract_variable_name(equation)
        self._variables = {var: sympy.symbols(var) for var in potential_variable
                           if var in variable_names}

    def _extract_variable_name(self, equation):
        variables = ''.join(character if character.isalpha()
                            else ' ' for character in equation).split()
        return variables

    def _generate_expression_input(self, all_input: dict):
        expression_input = {self._variables[var]: all_input[var]
                            for var in [*self._variables.keys()]}
        return expression_input

    def evaluate(self, x=None, y=None, z=None, t=None, **coupled_variables):
        full_input = {'x': x, 'y': y, 'z': z, 't': t} \
            | {variable_name: value for variable_name, value in coupled_variables.items()}
        expression_input = self._generate_expression_input(full_input)
        expression = self._expression.subs(expression_input)

        return float(expression)
