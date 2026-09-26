"""Shared enums for workflow, roles, messages, and jobs."""

from enum import Enum


class Stage(str, Enum):
    NEW = "NEW"
    PREPARATION = "PREPARATION"
    OPTIMIZATION = "OPTIMIZATION"
    DEPLOYMENT = "DEPLOYMENT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ResearchStage(str, Enum):
    PREPARATION = "PREPARATION"
    TRAINING_EVALUATION = "TRAINING_EVALUATION"
    DEPLOYMENT = "DEPLOYMENT"


class WorkflowStatus(str, Enum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    WAITING_FOR_USER = "waiting_for_user"
    WAITING_FOR_AGENT = "waiting_for_agent"
    WAITING_FOR_JOB = "waiting_for_job"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PreparationState(str, Enum):
    COLLECT_METADATA = "COLLECT_METADATA"
    VALIDATE_METADATA = "VALIDATE_METADATA"
    LOAD_DATASET = "LOAD_DATASET"
    CONFIRM_COLUMNS = "CONFIRM_COLUMNS"
    CHECK_REQUIRED_COLUMNS = "CHECK_REQUIRED_COLUMNS"
    ANALYZE_DATA_QUALITY = "ANALYZE_DATA_QUALITY"
    CONFIRM_CLEANING = "CONFIRM_CLEANING"
    APPLY_CLEANING = "APPLY_CLEANING"
    CONFIGURE_OBJECTIVE = "CONFIGURE_OBJECTIVE"
    PREPARATION_READY = "PREPARATION_READY"


class OptimizationState(str, Enum):
    INITIALIZE_SEARCH = "INITIALIZE_SEARCH"
    INITIAL_RANDOM_TRIALS = "INITIAL_RANDOM_TRIALS"
    SUMMARIZE_HISTORY = "SUMMARIZE_HISTORY"
    WAIT_FOR_OPTIONAL_USER_GUIDANCE = "WAIT_FOR_OPTIONAL_USER_GUIDANCE"
    MODEL_MANAGER_PLAN = "MODEL_MANAGER_PLAN"
    VALIDATE_GUIDANCE = "VALIDATE_GUIDANCE"
    GENERATE_TRIAL_BATCH = "GENERATE_TRIAL_BATCH"
    EXECUTE_BATCH = "EXECUTE_BATCH"
    UPDATE_EXPERIMENT_STORE = "UPDATE_EXPERIMENT_STORE"
    UPDATE_VISUALIZATIONS = "UPDATE_VISUALIZATIONS"
    CHECK_STOPPING_CONDITION = "CHECK_STOPPING_CONDITION"
    OPTIMIZATION_COMPLETE = "OPTIMIZATION_COMPLETE"


class DeploymentState(str, Enum):
    SELECT_BEST_CONFIGURATION = "SELECT_BEST_CONFIGURATION"
    BUILD_OR_LOAD_FINAL_MODEL = "BUILD_OR_LOAD_FINAL_MODEL"
    RUN_FORECAST = "RUN_FORECAST"
    BUILD_CONTEXT_VIEW = "BUILD_CONTEXT_VIEW"
    RETRIEVE_SIMILAR_DAYS = "RETRIEVE_SIMILAR_DAYS"
    WAIT_FOR_USER_ANALYSIS_OR_ADJUSTMENT = "WAIT_FOR_USER_ANALYSIS_OR_ADJUSTMENT"
    VALIDATE_POSTPROCESS_REQUEST = "VALIDATE_POSTPROCESS_REQUEST"
    APPLY_POSTPROCESSING = "APPLY_POSTPROCESSING"
    RECOMPUTE_METRICS_IF_LABELS_AVAILABLE = "RECOMPUTE_METRICS_IF_LABELS_AVAILABLE"
    SAVE_DEPLOYMENT_RESULT = "SAVE_DEPLOYMENT_RESULT"


class Actor(str, Enum):
    USER = "user"
    SYSTEM = "system"
    SERVICE = "service"
    TASK_MANAGER = "task_manager"
    PREPARATION_ASSISTANT = "preparation_assistant"
    MODEL_MANAGER = "model_manager"
    MODEL_DEVELOPER = "model_developer"
    DEPLOYMENT_OPERATOR = "deployment_operator"


AGENT_ROLES = frozenset(
    {
        Actor.TASK_MANAGER,
        Actor.PREPARATION_ASSISTANT,
        Actor.MODEL_MANAGER,
        Actor.MODEL_DEVELOPER,
        Actor.DEPLOYMENT_OPERATOR,
    }
)


class Topic(str, Enum):
    CHAT = "chat"
    PREPARE = "prepare"
    OPTIMIZE = "optimize"
    TRAIN = "train"
    DEPLOY = "deploy"
    VISUALIZATION = "visualization"
    SYSTEM = "system"


class MessageKind(str, Enum):
    USER = "user"
    COMMAND = "command"
    EVENT = "event"
    RESULT = "result"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrialStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationMode(str, Enum):
    VANILLA_BO = "vanilla_bo"
    LLM_GUIDED = "llm_guided"
    HUMAN_LLM_GUIDED = "human_llm_guided"


class ModelFamily(str, Enum):
    LINEAR = "Linear"
    SVR = "SVR"
    MLP = "MLP"
    XGBOOST = "XGBoost"
    LSTM = "LSTM"
    GRU = "GRU"
    CNN = "CNN"
