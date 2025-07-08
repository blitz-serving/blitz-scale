## How to better reproduce results

Internally, we use our bench runner (XX) to simplify future reproduction.
For each major feature related to performance,
we need to do the following steps:

1. Create a folder in the `exps` folder, whose name is related to the feature.
2. In the folder, create a `README.md` to record the hardware setup of the evaluation as well as step-by-step instructions on how to reproduce the results on the hardware setup.
3. In the folder, create a `run.toml` that others can use to reproduce the experiments with the bench runner.

Please refer to `exps/sample/` for a template for the `README.md` and `run.toml`.