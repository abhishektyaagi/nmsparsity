import pandas as pd
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("condensed-sparsity/condensed-rigl")

summary_list, config_list, name_list, state_list, id_list, tags = (
    [],
    [],
    [],
    [],
    [],
    [],
)
sweep_list = []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k, v in run.config.items() if not k.startswith("_")}
    )

    # .name is the human-readable name of the run.
    name_list.append(run.name)
    id_list.append(run.id)
    state_list.append(run.state)
    sweep_id = None
    if hasattr(run.sweep, "id"):
        sweep_id = run.sweep.id
    sweep_list.append(sweep_id)
    tags.append(run.tags)

runs_df = pd.DataFrame(
    {
        "summary": summary_list,
        "config": config_list,
        "name": name_list,
        "id": id_list,
        "state": state_list,
        "sweep_id": sweep_list,
        "tags": tags,
    }
)

runs_df.to_csv("project.csv")
