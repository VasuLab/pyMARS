import importlib.resources
from tempfile import TemporaryDirectory

import cantera as ct

from pymars.sampling import data_files, InputIgnition
from pymars.sensitivity_analysis import run_sa


def get_asset_path(filename: str) -> str:
    """Returns the file path to the requested asset file."""
    return str(importlib.resources.files('pymars.tests.assets').joinpath(filename))


def check_equal(list1, list2):
    """Check whether two lists have the same contents (regardless of order).

    Taken from https://stackoverflow.com/a/12813909

    Parameters
    ----------
    list1 : list
        First list, containing all of a particular type
    list2: list
        Second list, containing all of a particular type

    Returns
    -------
    bool
        ``True`` if lists are equal

    """
    return len(list1) == len(list2) and sorted(list1) == sorted(list2)


class TestRunSA:
    def test_drgepsa(self):
        """Test SA using stored DRGEP result with upper_threshold = 0.5
        """
        starting_model = get_asset_path('drgep_gri30.yaml')
        conditions = [
            InputIgnition(
                kind='constant volume', pressure=1.0, temperature=1000.0, equivalence_ratio=1.0,
                fuel={'CH4': 1.0}, oxidizer={'O2': 1.0, 'N2': 3.76}
                ),
            InputIgnition(
                kind='constant volume', pressure=1.0, temperature=1200.0, equivalence_ratio=1.0,
                fuel={'CH4': 1.0}, oxidizer={'O2': 1.0, 'N2': 3.76}
                ),
        ]
        data_files['output_ignition'] = get_asset_path('example_ignition_output.txt')
        data_files['data_ignition'] = get_asset_path('example_ignition_data.dat')
        
        limbo_species = ['H2', 'H2O2', 'CH2(S)', 'C2H4', 'C2H5', 'C2H6', 'HCCO', 'CH2CO']

        # Get expected model	
        expected_model = ct.Solution(get_asset_path('drgepsa_gri30.yaml'))

        # try using initial SA method
        with TemporaryDirectory() as temp_dir:
            reduced_model = run_sa(
                starting_model, 3.22, conditions, [], [], 5.0, ['N2'], 
                algorithm_type='initial', species_limbo=limbo_species[:], num_threads=1, 
                path=temp_dir
                )

        # Make sure models are the same	
        assert check_equal(reduced_model.model.species_names, expected_model.species_names)
        assert reduced_model.model.n_reactions == expected_model.n_reactions
        assert round(reduced_model.error, 2) == 3.20

        # try using greedy SA method
        with TemporaryDirectory() as temp_dir:
            reduced_model = run_sa(
                starting_model, 3.22, conditions, [], [], 5.0, ['N2'], 
                algorithm_type='greedy', species_limbo=limbo_species[:], num_threads=1, 
                path=temp_dir
                )
        
        # Make sure models are the same	
        assert check_equal(reduced_model.model.species_names, expected_model.species_names)
        assert reduced_model.model.n_reactions == expected_model.n_reactions
        assert round(reduced_model.error, 2) == 3.20
