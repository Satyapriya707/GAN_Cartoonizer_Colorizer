from pydantic import BaseModel
from typing import List
from enum import Enum

class dropdownChoices(str, Enum):
    hosoda = "Hosoda"
    hayao = "Hayao"
    shinkai = "Shinkai"
    paprika = "Paprika"
    all = "All"

class dropdownChoicesSingleImage(str, Enum):
    hosoda = "Hosoda"
    hayao = "Hayao"
    shinkai = "Shinkai"
    paprika = "Paprika"

class dropdownChoicesWB(str, Enum):
    type_0 = "Type-0"
    type_1 = "Type-1"
    type_2 = "Type-2"
    type_3 = "Type-3"
    all = "All"

class dropdownChoicesSingleImageWB(str, Enum):
    type_0 = "Type-0"
    type_1 = "Type-1"
    type_2 = "Type-2"
    type_3 = "Type-3"
