from flowforge.parsers.EquationParser import EquationParser

def test_constant_equation():
    test_name = "test_constant_equation"
    equation = '11.2'
    expression = EquationParser(equation)
    variables = ['x']

    test_inputs = [[1],
                   [5],
                   [1.2]]
    expected_results = [11.2, 11.2, 11.2]

    test_all_values(variables, expression, test_name, equation,
                    test_inputs, expected_results)

def test_spatial_equation():
    test_name = "test_spatial_equation"
    equation = '10.1*x + (15.6 / y) - z**2'
    expression = EquationParser(equation)
    variables = ['x', 'y', 'z']

    test_inputs = [(1, 1, 1),
                   (5, 5, 5),
                   (1.2, 2.0, 0.2)]
    expected_results = [24.7, 28.62, 19.88]

    test_all_values(variables, expression, test_name, equation,
                    test_inputs, expected_results)

def test_time_dependent_equation():
    test_name = "test_time_dependent_equation"
    equation = '10*t'
    expression = EquationParser(equation)
    variables = ['t']

    test_inputs = [[10],
                   [20],
                   [30.42]]
    expected_results = [100, 200, 304.2]

    test_all_values(variables, expression, test_name, equation,
                    test_inputs, expected_results)

def test_spatial_and_time_equation():
    test_name = "test_spatial_and_time_equation"
    equation = '(10.1*x + (15.6 / y) - z**2) / t'
    expression = EquationParser(equation)
    variables = ['x', 'z', 'y', 't']

    test_inputs = [(1, 1, 1, 1),
                   (5, 5, 5, 5),
                   (1.2, 2.0, 0.2, 0.8)]
    expected_results = [24.7, 5.724, 24.85]

    test_all_values(variables, expression, test_name, equation,
                    test_inputs, expected_results)

def test_coupled_variables():
    test_name = "test_coupled_variables"
    equation = '(11.9*x + 3.6*z) * 13.2*temperature + 6*pressure'
    expression = EquationParser(equation, 'temperature', 'pressure')
    variables = ['x', 'z', 'temperature', 'pressure']

    test_inputs = [(1, 1, 1, 1),
                   (5, 5, 5, 5),
                   (1.2, 2.0, 0.2, 0.8)]
    expected_results = [210.6, 5145, 61.5072]

    test_all_values(variables, expression, test_name, equation,
                    test_inputs, expected_results)

def test_all_values(variables, expression, test_name, original_equation, test_inputs, expected_values):
    number_of_variables = len(variables)
    number_of_tests = len(test_inputs)

    inputs_and_expected = [(test_inputs[i], expected_values[i]) for i in range(number_of_tests)]
    for input, expected in inputs_and_expected:
        input_args = {variables[i] : input[i] for i in range(number_of_variables)}
        output_value = expression.evaluate(**input_args)
        compare_results(test_name, original_equation, input,
                        output_value, expected)

def compare_results(test_name, equation, input_values, output_value, expected_value):
    if withinRoundingPrecision(expected_value, output_value):
        return True
    else:
        raise Exception("ERROR in test " + test_name + " : " + \
                         "Expected " + str(equation) + " @ " + str(input_values) + \
                         "to give '"+str(expected_value)+"'. Got '"+str(output_value)+"' instead.")

def withinRoundingPrecision(true, predicted):
    PRECISION_FACTOR = 1e-10
    if (true - predicted) < PRECISION_FACTOR:
        return True
    else:
        return False

if __name__ == "__main__":
    test_constant_equation()
    test_spatial_equation()
    test_time_dependent_equation()
    test_spatial_and_time_equation()
    test_coupled_variables()