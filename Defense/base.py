from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class BaseDefenderConfig:
    pass

class BaseDefender:
    pass

    def get_response(self, prompts_list):
        pass
