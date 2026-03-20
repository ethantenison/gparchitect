# gparchitect
GPArchitect builds Gaussian Process models from natural-language instructions and tabular data using BoTorch and GPyTorch. Provide a pandas DataFrame and a text specification, and it constructs, fits, and validates SingleTaskGP, MultiTaskGP, or ModelListGP models. If fitting fails, it revises the instructions, retries, and logs all changes.
