#!/usr/bin/env python3

import adarl.utils.utils
from pathlib import Path
import sys

# leg_file = adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf")
model_definition_string = adarl.utils.utils.compile_xacro_string(  model_definition_string=Path(sys.argv[1]).read_text(),
                                                                    model_kwargs={})
print(model_definition_string)