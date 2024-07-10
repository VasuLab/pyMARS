""" Tests the create trimmed model unit used by pyMARS """

import os
import pkg_resources

import pytest
import cantera as ct

from ..reduce_model import trim

def relative_location(file):
    file_path = os.path.join(file)
    return pkg_resources.resource_filename(__name__, file_path)


class TestTrim:
    def test_GRI_minus_three(self):
        """Tests removal of three species from GRI Mech 3.0.
        """
        # Original model to remove things from
        initial_model = 'gri30.yaml'

        # Create exclusion list for test case
        exclusion_list = ["CH4", "O2", "N2"]

        # Run trim unit
        reduced_model = trim(initial_model, exclusion_list, 'gri30.yaml')

        # Expected answer	
        expected_species_num = 50
        expected_reactions_num = 237

        # Make sure number matches what is expected
        assert reduced_model.n_species == expected_species_num
        assert reduced_model.n_reactions == expected_reactions_num

        # Make sure removed species are not included
        assert "CH4" not in reduced_model.species_names
        assert "O2" not in reduced_model.species_names
        assert "N2" not in reduced_model.species_names

    def test_GRI_minus_zero(self):
        """Tests removal of zero species from GRI Mech 3.0.
        """
        # Original model to remove things from
        initial_model = 'gri30.yaml'

        # Create exclusion list for test case
        exclusion_list = []

        # Run trim unit
        reduced_model = trim(initial_model, exclusion_list, 'reduced_gri30.yaml')

        # Expected answer	
        expected_species_num = 53
        expected_reactions_num = 325

        # Make sure number matches what is expected
        assert reduced_model.n_species == expected_species_num
        assert reduced_model.n_reactions == expected_reactions_num
        assert reduced_model.name == 'reduced_gri30'

    def test_artificial_minus_one(self):
        """Test removing one species from artificial model.
        """
        # Original model to remove things from
        initial_model = relative_location(os.path.join('assets', 'artificial-mechanism.yaml'))

        # Create exclusion list for test case
        exclusion_list = ['H']

        # Run trim unit
        reduced_model = trim(initial_model, exclusion_list, 'a-m.yaml')

        # Expected answer	
        expected_species_num = 3
        expected_reactions_num = 1

        # Make sure number matches what is expected
        assert reduced_model.n_species == expected_species_num
        assert reduced_model.n_reactions == expected_reactions_num

        # Make sure removed species are not included
        assert 'H' not in reduced_model.species_names
        for sp in exclusion_list:
            assert all([sp not in {**rxn.reactants, **rxn.products} for rxn in reduced_model.reactions()])
        
    def testArtRemoveAll(self):
        """Test removing all four species in an artificial model.

        Raises exception because Cantera will not produce a Solution with no species/reactions.
        """

        # Original model to remove things from
        initial_model = relative_location(os.path.join('assets', 'artificial-mechanism.yaml'))

        # Create exclusion list for test case
        exclusion_list = ["H", "H2", "O2", "H2O"]

        with pytest.raises(ValueError):
            reduced_model = trim(initial_model, exclusion_list, "a-m.yaml")

    def testArtRemoveInvalid(self):
        """Test removing species not present in model.
        """
        # Original model to remove things from
        initial_model = relative_location(os.path.join('assets', 'artificial-mechanism.yaml'))

        # Create exclusion list for test case
        exclusion_list = ['CH4']

        # Run trim unit
        reduced_model = trim(initial_model, exclusion_list, 'a-m.yaml')

        # Expected answer	
        expected_species_num = 4
        expected_reactions_num = 2

        # Make sure number matches what is expected
        assert reduced_model.n_species == expected_species_num
        assert reduced_model.n_reactions == expected_reactions_num
        for sp in exclusion_list:
            assert all([sp not in {**rxn.reactants, **rxn.products} for rxn in reduced_model.reactions()])

    def testArtRemoveInvalidAnd1(self):
        """Test removing mixture of species both in and not in artificial model.
        """
        # Original model to remove things from
        initial_model = relative_location(os.path.join('assets', 'artificial-mechanism.yaml'))

        # Create exclusion list for test case
        exclusion_list = ["H", "CH4"]

        # Run trim unit
        reduced_model = trim(initial_model, exclusion_list, "a-m.yaml")

        # Expected answer	
        expected_species_num = 3
        expected_reactions_num = 1

        # Make sure number matches what is expected
        assert reduced_model.n_species == expected_species_num
        assert reduced_model.n_reactions == expected_reactions_num

        # Make sure removed species are not included
        assert "H" not in reduced_model.species_names
        for sp in exclusion_list:
            assert all([sp not in {**rxn.reactants, **rxn.products} for rxn in reduced_model.reactions()])

    def test_GRI_minus_10(self):
        """Test removing 10 species from GRI Mech 3.0
        """
        # Original model to remove things from
        initial_model = 'gri30.yaml'

        # Create exclusion list for test case
        exclusion_list = ["CH4", "O2", "N2", "H", "OH", "H2O", "CH2", "CH3", "CO", "AR"]

        # Run trim unit
        reduced_model = trim(initial_model, exclusion_list, 'reduced_gri30.yaml')

        # Expected answer	
        expected_species_num = 43
        expected_reactions_num = 14

        # Make sure number matches what is expected
        assert reduced_model.n_species == expected_species_num
        assert reduced_model.n_reactions == expected_reactions_num

        # Make sure removed species are not included
        for sp in exclusion_list:
            assert sp not in reduced_model.species_names
            assert all([sp not in {**rxn.reactants, **rxn.products} for rxn in reduced_model.reactions()])
    
    def test_remove_explicit_third_bodies(self):
        """Tests appropriate removal of reactions with explicit third body species.
        """
        initial_model = relative_location(os.path.join('assets', 'model-third-bodies.yaml'))
        reduced_model = trim(initial_model, ['ar', 'he'], 'test.yaml')

        assert reduced_model.n_species == 4
        assert reduced_model.n_reactions == 1
