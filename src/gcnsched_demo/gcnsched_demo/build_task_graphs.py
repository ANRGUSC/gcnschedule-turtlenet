import pathlib
from wfcommons.common.workflow import Workflow
from wfcommons.wfchef.recipes.cycles.recipe import CyclesRecipe
from wfcommons.wfchef.recipes.montage import MontageRecipe
from wfcommons.wfchef.recipes.seismology import SeismologyRecipe
from wfcommons.wfchef.recipes.blast import BlastRecipe
from wfcommons.wfchef.recipes.bwa import BwaRecipe
from wfcommons.wfchef.recipes.epigenomics import EpigenomicsRecipe
from wfcommons.wfchef.recipes.srasearch import SrasearchRecipe
from wfcommons.wfchef.recipes.genome import GenomeRecipe
from wfcommons.wfchef.recipes.soykb import SoykbRecipe
from wfcommons.wfchef.utils import draw

import pathlib
import json

RECIPES = {
    "montage": MontageRecipe,
    "cycles": CyclesRecipe,
    "seismology": SeismologyRecipe,
    "blast": BlastRecipe,
    "bwa": BwaRecipe,
    "epigenomics": EpigenomicsRecipe,
    "srasearch": SrasearchRecipe,
    "genome": GenomeRecipe,
    "soykb": SoykbRecipe
}

RECIPE = "epigenomics"
NUM_TASKS = 43

def main():
    recipe = RECIPES[RECIPE](num_tasks=NUM_TASKS) 
    workflow: Workflow = recipe.build_workflow("my_workflow") 

if __name__ == "__main__":
    main()