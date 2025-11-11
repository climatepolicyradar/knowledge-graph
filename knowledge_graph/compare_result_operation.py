from pydantic import BaseModel

from flows.classifier_specs.spec_interface import ClassifierSpec
from knowledge_graph.classifiers_profiles import ClassifiersProfileMapping


class CompareResultOperation(BaseModel):
    """Base class for compare result operation"""


class Promote(CompareResultOperation):
    """Class to represent promoting a classifier profile mapping"""

    classifiers_profile_mapping: ClassifiersProfileMapping


class Demote(CompareResultOperation):
    """Class to represent demoting classifier specs"""

    classifier_spec: ClassifierSpec


class Update(CompareResultOperation):
    """Class to represent updating classifier specs with classifiers profile mapping"""

    classifier_spec: ClassifierSpec
    classifiers_profile_mapping: ClassifiersProfileMapping


class Ignore(CompareResultOperation):
    """Class to represent ignoring a classifier spec"""

    classifier_spec: ClassifierSpec
