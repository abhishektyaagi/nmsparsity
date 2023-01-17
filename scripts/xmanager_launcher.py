# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XManager launcher that runs an image built from a Dockerfile."""

from typing import Sequence

from absl import app
from xmanager import xm
from xmanager import xm_local
from dotenv import dotenv_values


def main(argv: Sequence[str]) -> None:
    del argv

    with xm_local.create_experiment(
        experiment_title="condensed-sparsity-docker-test"
    ) as experiment:
        executable_spec = xm.Dockerfile(
            path="/home/mike/condensed-sparsity/",
            dockerfile="/home/mike/condensed-sparsity/Dockerfile.gcs",
        )
        # executable_spec = xm.Container("gcs:latest")

        [executable] = experiment.package(
            [
                xm.Packageable(
                    executable_spec=executable_spec,
                    executor_spec=xm_local.Vertex.Spec(),
                    args={},
                    env_vars={},
                ),
            ]
        )
        env_vars = dotenv_values("/home/mike/condensed-sparsity/.env.gcs")
        experiment.add(
            xm.Job(
                executable=executable,
                executor=xm_local.Vertex(),
                env_vars=env_vars,
                args=[],
            )
        )


if __name__ == "__main__":
    app.run(main)
