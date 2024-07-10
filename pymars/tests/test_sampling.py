""" Tests the sampling module in pyMARS """

import pytest

from pymars.sampling import parse_ignition_inputs, InputIgnition


class TestCheckInputs:
    def test_good_example(self):
        """Tests (good) sample input file.
        """
        inputs = [
            {'kind': 'constant volume', 'pressure': 1.0, 'temperature': 1000.0,
             'fuel': {'CH4': 1.0}, 'oxidizer': {'O2': 1.0, 'N2': 3.76}, 'equivalence-ratio': 1.0},
            {'kind': 'constant volume', 'pressure': 1.0, 'temperature': 1200.0,
             'fuel': {'CH4': 1.0}, 'oxidizer': {'O2': 1.0, 'N2': 3.76}, 'equivalence-ratio': 1.0}
             ]
        
        conditions = parse_ignition_inputs('gri30.yaml', inputs)
        for item in conditions:
            assert type(item) == InputIgnition
    
    def test_good_example_alternate(self):
        """Tests correct sample input file with alternate values.
        """
        inputs = [
            {'kind': 'constant pressure', 'pressure': 1.0, 'temperature': 1000.0,
             'end time': 10.0, 'reactants': {'CH4': 1.0, 'O2': 1.0, 'N2': 3.76}},
             ]
        
        conditions = parse_ignition_inputs('gri30.yaml', inputs)
        for item in conditions:
            assert type(item) == InputIgnition
    
    @pytest.mark.parametrize('key', [
        'kind', 'pressure', 'temperature', 'fuel', 'oxidizer', 'equivalence-ratio'
        ])
    def test_missing_keys(self, key):
        """Tests correct errors for missing required keys.
        """
        case = [{
            'kind': 'constant volume',
            'pressure': 1.0,
            'temperature': 1000.0,
            'end-time': 10.0,
            'fuel': {'CH4': 1.0},
            'oxidizer': {'O2': 1.0, 'N2': 3.76},
            'equivalence-ratio': 1.0
            }]
        del case[0][key]

        with pytest.raises(AssertionError):
            parse_ignition_inputs('gri30.yaml', case)
    
    def test_bad_fuel_oxidizer_value(self):
        """Tests correct errors for improper value.
        """
        case = [{
            'kind': 'constant volume',
            'pressure': 1.0,
            'temperature': 1000,
            'end-time': 10.0,
            'fuel': {'CH4': 0.0},
            'oxidizer': {'O2': 1.0, 'N2': 3.76},
            'equivalence-ratio': 1.0
            }]
        with pytest.raises(AssertionError):
            parse_ignition_inputs('gri30.yaml', case)
        
        case = [{
            'kind': 'constant volume',
            'pressure': 1.0,
            'temperature': 1000,
            'end-time': 10.0,
            'fuel': {'CH4': 1.0},
            'oxidizer': {'O2': 0.0, 'N2': 3.76},
            'equivalence-ratio': 1.0
            }]
        with pytest.raises(AssertionError):
            parse_ignition_inputs('gri30.yaml', case)
    
    def test_bad_species(self):
        """Tests raising error for species not in model.
        """
        case = [{
            'kind': 'constant volume',
            'pressure': 1.0,
            'temperature': 1000,
            'end-time': 10.0,
            'fuel': {'C4H10': 0.0},
            'oxidizer': {'O2': 1.0, 'N2': 3.76},
            'equivalence-ratio': 1.0
            }]
        with pytest.raises(AssertionError):
            parse_ignition_inputs('gri30.yaml', case)
        
        case = [{
            'kind': 'constant volume',
            'pressure': 1.0,
            'temperature': 1000,
            'end-time': 10.0,
            'fuel': {'CH4': 1.0},
            'oxidizer': {'O2': 0.0, 'HE': 3.76},
            'equivalence-ratio': 1.0
            }]
        with pytest.raises(AssertionError):
            parse_ignition_inputs('gri30.yaml', case)

        case = [{
            'kind': 'constant volume',
            'pressure': 1.0,
            'temperature': 1000,
            'end-time': 10.0,
            'reactants': {'C4H10': 1.0, 'O2': 1.0, 'N2': 3.76}
            }]
        with pytest.raises(AssertionError):
            parse_ignition_inputs('gri30.yaml', case)
