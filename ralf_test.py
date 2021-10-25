import ralf
import argparse
from redis import Redis
import pickle
from pathlib import Path
import json
import os
import time
from typing import Dict, List

from absl import flags, app

import ray
import redis
from ralf.core import Ralf
from ralf.operator import DEFAULT_STATE_CACHE_SIZE, Operator
from ralf.operators.source import Source
from ralf.policies import load_shedding_policy, processing_policy
from ralf.state import Record, Schema
from ralf.table import Table
from statsmodels.tsa.seasonal import STL
from absl import flags, app
from ralf.table import Table

window_size = 672
global_slide_size = 48
per_key_slide_size_plan_path = ""
SEASONALITY = 48
seasonality = 48
redis_model_db_id = 2
experiment_dir = None
experiment_id = ""
log_wandb = False


@ray.remote
class RedisSource(Source):
    def __init__(
        self,
        num_worker_threads: int = 1,
    ):
        schema = Schema(
            "key",
            {
                "key": str,
                "value": float,
                "timestamp": int,
                "send_time": float,
                "create_time": float,
            },
        )
        super().__init__(schema, num_worker_threads=num_worker_threads)
        self.consumer = redis.Redis()

    def next(self) -> List[Record]:
        _, data = self.consumer.xreadgroup(
            "ralf-reader-group",
            f"reader-{self._shard_idx}",
            {"ralf": ">"},
            count=1,
            block=100 * 1000,
        )[0]
        record_id, payload = data[0]
        self.last_record = record_id
        self.consumer.xack("ralf", "ralf-reader-group", record_id)

        assert isinstance(payload, dict)
        record = Record(
            key=payload[b"key"].decode(),
            value=payload[b"value"].decode(),
            timestamp=int(payload[b"timestamp"]),
            send_time=float(payload[b"send_time"]),
            create_time=time.time(),
        )
        return [record]
@ray.remote
class RedisSink(Operator):
    def __init__(self, db_id: int, num_worker_threads: int = 1):
        super().__init__(
            schema=None,
            num_worker_threads=num_worker_threads,
        )
        self.redis_conn = Redis(db=db_id)

    def on_record(self, record: Record):
        self.redis_conn.set(record.key, pickle.dumps(record))
        
@ray.remote
class DummyTrainer(Operator):
    def __init__(
        self,
        seasonality,
        num_worker_threads=4,
        processing_policy=processing_policy.last_completed,
        load_shedding_policy=load_shedding_policy.later_complete_time,
    ):
        super().__init__(
            schema,
            num_worker_threads=num_worker_threads,
            processing_policy=processing_policy,
            load_shedding_policy=load_shedding_policy,
        )
    def on_record(self, record: Record) -> Record:
        return Record(
            key=record.key,
            trend=0,
            seasonality=[0,0,0],
            create_time=record.create_time,
            complete_time=time.time(),
            timestamp=record.timestamp,
        )


def create_test_pipeline(metrics_dir):

    # create Ralf instance
    ralf_conn = Ralf(
        metric_dir=os.path.join("exp_dir"),
        log_wandb=log_wandb,
        exp_id=experiment_id,
    )

    source = Table(
        [],
        RedisSource,
        num_replicas=1,
        num_worker_threads=1,
    )

    (
        source.window(
            window_size,
            global_slide_size,
            num_replicas=1,
            num_worker_threads=1,
            per_key_slide_size_plan_file=per_key_slide_size_plan_path,
        )
        .map(
            DummyTrainer,
            seasonality,
            num_replicas=1,
            num_worker_threads=1,
        )
        .map(
            RedisSink,
            num_replicas=1,
            num_worker_threads=1,
            db_id=redis_model_db_id,
        )
    )

    # TODO(simon): user should be able to deploy pipeline as a whole, just deploying source is hard to use.
    # deploy
    ralf_conn.deploy(source, "source")

    return ralf_conn

FLAGS = flags.FLAGS
# flags.DEFINE_integer("window_size", 672, "window size for stl trainer")
# flags.DEFINE_integer("global_slide_size", 48, "static slide size configuration")
# flags.DEFINE_string(
#     "per_key_slide_size_plan_path", "", "pre key slide size configuration"
# )
# flags.DEFINE_integer("seasonality", 48, "seasonality for stl training process")
# flags.DEFINE_integer(
#     "redis_model_db_id", default=2, help="Redis DB number for db snapshotting."
# )

# flags.DEFINE_string(
#     "experiment_dir", None, "directory to write metadata to", required=True
# )
# flags.DEFINE_string("experiment_id", "", "experiment run name for wandb")
# flags.DEFINE_bool("log_wandb", False, "whether to use wandb logging")



def _get_config() -> Dict:
    """Return all the flag vlaue defined here."""
    return {f.name: f.value for f in FLAGS.get_flags_for_module("__main__")}



def main(argv):
    print("Using config", _get_config())
    exp_dir = Path("exp_dir")
    exp_dir.mkdir(exist_ok=True, parents=True)
    with open(exp_dir / "server_config.json", "w") as f:
        json.dump(_get_config(), f)
    ralf_conn = create_test_pipeline(exp_dir / "metrics")
    ralf_conn.run()


if __name__ == "__main__":
    app.run(main)
