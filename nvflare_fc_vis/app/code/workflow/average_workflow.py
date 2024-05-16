from nvflare.apis.impl.controller import Controller, Task, ClientTask
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.shareable import Shareable

import os
import json


class AverageWorkflow(Controller):
    def __init__(
        self,
        aggregator_id="aggregator",
        min_clients: int = 2,
        num_rounds: int = 2,
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        train_timeout: int = 0,
        ignore_result_error: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
        snapshot_every_n_rounds: int = 1,
    ):
        super().__init__()
        self.aggregator_id = aggregator_id
        self.aggregator = None
        self._train_timeout = train_timeout
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._start_round = start_round
        self._wait_time_after_min_received = wait_time_after_min_received
        self._ignore_result_error = ignore_result_error
        self._task_check_period = task_check_period
        self._persist_every_n_rounds = persist_every_n_rounds
        self._snapshot_every_n_rounds = snapshot_every_n_rounds
        pass

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.aggregator = self._engine.get_component(self.aggregator_id)

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        fl_ctx.set_prop(key="CURRENT_ROUND", value=0)

        # load parameters.json and set to the context that will be shared with clients
        parameters_file_path = self.get_parameters_file_path()
        computation_parameters = self.load_computation_parameters(parameters_file_path)

        fl_ctx.set_prop(key="COMPUTATION_PARAMETERS", value=computation_parameters, private=False, sticky=True)

        # create the initial task
        get_local_average_task = Task(
            name="get_local_average_and_count",
            data=Shareable(),
            props={},
            timeout=self._train_timeout,
            # before_task_sent_cb=self._prepare_train_task_data,
            result_received_cb=self._accept_site_result,
        )

        # broadcast the task to all clients and await their responses
        self.broadcast_and_wait(
            task=get_local_average_task,
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        # once the all responses are returned, start the aggregation process
        self.log_info(fl_ctx, "Start aggregation.")
        aggr_shareable = self.aggregator.aggregate(fl_ctx)
        self.log_info(fl_ctx, "End aggregation.")

        result = {"global_average": aggr_shareable.get("global_average", {})}
        print(f"\n\n{'='*50}\nAggregated result: {result}\n{'='*50}\n\n")

        # create a task to accept the global average
        accept_global_average_task = Task(
            name="accept_global_average",
            data=aggr_shareable,
            props={},
            timeout=self._train_timeout,
        )

        # broadcast the global average to all clients
        self.broadcast_and_wait(
            task=accept_global_average_task,
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

    def _accept_site_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        accepted = self.aggregator.accept(client_task.result, fl_ctx)
        return accepted

    def process_result_of_unknown_task(self, task: Task, fl_ctx: FLContext) -> None:
        pass

    def get_parameters_file_path(self) -> str:
        """
        Determines the appropriate data directory path for the federated learning application by checking
        if in production, simulator or poc mode.
        """

        production_path = os.getenv("PARAMETERS_FILE_PATH")
        simulator_path = os.path.abspath(os.path.join(os.getcwd(), "../test_data", "server", "parameters.json"))
        poc_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../test_data", "server", "parameters.json"))

        print("\n\n")
        print(f"production_path: {production_path}")
        print(f"simulator_path: {simulator_path}")
        print(f"poc_path: {poc_path}")
        print("\n\n")
        
        if production_path:
            return production_path
        if os.path.exists(simulator_path):
            return simulator_path
        elif os.path.exists(poc_path):
            return poc_path
        else:
            raise FileNotFoundError(
                "parameters file path could not be determined.")

    def load_computation_parameters(self, parameters_file_path: str):
        with open(parameters_file_path, "r") as file:
            return json.load(file)
