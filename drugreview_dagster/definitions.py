
from dagster import Definitions
from .notebook_runner import notebook_job

defs = Definitions(jobs=[notebook_job])
