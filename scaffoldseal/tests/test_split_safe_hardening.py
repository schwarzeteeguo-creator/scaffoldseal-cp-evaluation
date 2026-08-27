from dataclasses import replace
import functools
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from split_safe import (  # noqa: E402
    FitAuditTrail,
    FramePredictionOutput,
    FreshTrainingState,
    OuterFoldContract,
    SafeFitOptions,
    SplitSafeFitExecutor,
    SplitSafePreprocessor,
    SplitViolation,
    canonical_id_hash,
    fit_boundary_policy,
)


ZERO_HASH = "0" * 64
ONE_HASH = "1" * 64
TWO_HASH = "2" * 64
CLASS_PAYLOAD = None


def valid_frame_prediction_output(
    state, outer_test_frame, feature_columns, telemetry=None
):
    ids = tuple(outer_test_frame["curated_id"].astype(str))
    predictions = pd.Series(
        outer_test_frame[feature_columns[0]].to_numpy(float) + state.model.offset,
        index=pd.Index(ids, name="curated_id"),
        name="prediction",
    )
    return FramePredictionOutput(predictions, telemetry or ())


def tiny_contract() -> OuterFoldContract:
    return OuterFoldContract(
        1,
        {"T1", "T2", "T3", "T4"},
        {"X1"},
        {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
    )


def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "curated_id": ["T1", "T2", "T3", "T4", "X1"],
            "feature": [1.0, 2.0, 3.0, 4.0, 999.0],
            "feature_2": [2.0, 4.0, 6.0, 8.0, -999.0],
            "target": [1.0, 2.0, 3.0, 4.0, -999.0],
            "permeability": [11.0, 12.0, 13.0, 14.0, -9999.0],
            "log_Papp": [21.0, 22.0, 23.0, 24.0, -9999.0],
            "replicate_spread": [0.1, 0.2, 0.3, 0.4, 9999.0],
            "source": ["S1", "S2", "S3", "S4", "SX"],
            "weight": [1.0, 2.0, 3.0, 4.0, 999.0],
        }
    )


class SpyEstimator:
    model_config_sha256 = ONE_HASH

    def __init__(self):
        self.fit_calls = 0
        self.fit_ids = ()
        self.eval_ids = ()
        self.prediction_offset = 0.0

    def fit(self, X, y, **kwargs):
        self.fit_calls += 1
        self.fit_ids = tuple(X.index.astype(str))
        if "eval_set" in kwargs:
            validation_X, _ = kwargs["eval_set"][0]
            self.eval_ids = tuple(validation_X.index.astype(str))
        return self

    def predict(self, X):
        return X["feature"].to_numpy(float) + float(self.prediction_offset)


class ClassPayloadEstimator(SpyEstimator):
    outer_payload = None


class CountingFactory:
    def __init__(self):
        self.calls = 0
        self.model_config_sha256 = ONE_HASH

    def __call__(self):
        self.calls += 1
        return SpyEstimator()


class StateBox:
    def __init__(self):
        self.offset = 0.0


class PredictionSeriesSubclass(pd.Series):
    @property
    def _constructor(self):
        return PredictionSeriesSubclass


class StateDictBox:
    def __init__(self):
        self.offset = 0.0
        self.parameter = np.asarray([1.0, 2.0], dtype=np.float64)
        self.buffer = np.asarray([3.0], dtype=np.float32)

    def state_dict(self):
        return {"weight": self.parameter, "running_buffer": self.buffer}


class SpyFrameAdapter:
    transform_sha256 = ZERO_HASH
    fitted_transform_sha256 = ZERO_HASH
    model_config_sha256 = ONE_HASH
    checkpoint_sha256 = TWO_HASH

    def __init__(self, best_epoch=2):
        self.best_epoch = int(best_epoch)
        self.create_calls = 0
        self.inner_calls = 0
        self.outer_calls = 0
        self.outer_predict_calls = 0
        self.last_train_ids = ()
        self.last_validation_ids = ()
        self.last_outer_columns = ()
        self.last_fixed_epoch = 0
        self.last_predict_columns = ()
        self.last_outer_predict_ids = ()

    def create_fresh_state(self, *, run_context):
        self.create_calls += 1
        return FreshTrainingState(StateBox(), StateBox(), StateBox())

    def fit_inner(
        self,
        *,
        state,
        train_frame,
        validation_frame,
        target_column,
        feature_columns,
        sample_weight,
        maximum_epochs,
        options,
        run_context,
        recorder,
    ):
        self.inner_calls += 1
        self.last_train_ids = tuple(train_frame["curated_id"].astype(str))
        self.last_validation_ids = tuple(validation_frame["curated_id"].astype(str))
        for epoch in range(1, maximum_epochs + 1):
            state.model.offset = float(abs(epoch - self.best_epoch))
            recorder.evaluate_frame_epoch(epoch, self, state)

    def fit_outer(
        self,
        *,
        state,
        train_frame,
        target_column,
        feature_columns,
        sample_weight,
        fixed_epoch,
        options,
        run_context,
    ):
        self.outer_calls += 1
        self.last_train_ids = tuple(train_frame["curated_id"].astype(str))
        self.last_outer_columns = tuple(train_frame.columns)
        self.last_fixed_epoch = int(fixed_epoch)

    def predict_validation(
        self,
        *,
        state,
        validation_frame,
        feature_columns,
        run_context,
    ):
        self.last_predict_columns = tuple(validation_frame.columns)
        return validation_frame[feature_columns[0]].to_numpy(float) + state.model.offset

    def predict_outer(
        self,
        *,
        state,
        outer_test_frame,
        feature_columns,
        run_context,
    ):
        columns = tuple(outer_test_frame.columns)
        if columns != ("curated_id", *feature_columns):
            raise AssertionError("Outer predictor received undeclared metadata")
        ids = tuple(outer_test_frame["curated_id"].astype(str))
        predictions = pd.Series(
            outer_test_frame[feature_columns[0]].to_numpy(float) + state.model.offset,
            index=pd.Index(ids, name="curated_id"),
            name="prediction",
        )
        return FramePredictionOutput(
            predictions,
            {
                "outer_predict_calls": 1,
                "run_object_id": id(run_context),
            },
        )


class AlternateSpyFrameAdapter(SpyFrameAdapter):
    pass


class ReusingFrameAdapter(SpyFrameAdapter):
    def __init__(self):
        super().__init__()
        self.shared_model = StateBox()
        self.shared_optimizer = StateBox()
        self.shared_scheduler = StateBox()

    def create_fresh_state(self, *, run_context):
        self.create_calls += 1
        return FreshTrainingState(
            self.shared_model, self.shared_optimizer, self.shared_scheduler
        )


class ParentSubset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [0]


class PayloadFrameAdapter(SpyFrameAdapter):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload


class SplitSafeHardeningTests(unittest.TestCase):
    def _recorder(
        self,
        contract,
        train,
        validation,
        run,
        audit,
        *,
        feature_columns=("feature",),
        target_column="target",
        transform_sha256=ZERO_HASH,
        model_config_sha256=ONE_HASH,
    ):
        return contract.create_inner_evaluation_recorder(
            train,
            validation,
            basket=int(train.inner_basket),
            feature_columns=feature_columns,
            target_column=target_column,
            metric_identity="mean_absolute_error",
            run_context=run,
            transform_sha256=transform_sha256,
            model_config_sha256=model_config_sha256,
            checkpoint_sha256=TWO_HASH,
            audit=audit,
        )

    def test_estimator_rejects_eval_set_callbacks_nested_data_before_factory(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        train = contract.inner_training_batch(frame, 1)
        validation = contract.inner_validation_batch(frame, 1)
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="attack", seed=0, inner_basket=1
            )
            factory = CountingFactory()
            attacks = [
                {"eval_set": [(frame.loc[frame.curated_id == "X1"], [999.0])]},
                {"callbacks": [lambda: frame]},
                {"verbose": {"callback": frame.loc[frame.curated_id == "X1"]}},
                {"verbose": frame},
                {"verbose": frame["target"]},
                {"verbose": np.asarray([1.0, 2.0])},
                {"verbose": [True]},
            ]
            for options in attacks:
                with self.subTest(options=tuple(options)):
                    with self.assertRaises(SplitViolation):
                        executor.fit_inner_estimator(
                            factory,
                            train,
                            validation,
                            basket=1,
                            feature_columns=["feature"],
                            target_column="target",
                            run_context=run,
                            options=options,
                        )
            self.assertEqual(factory.calls, 0)
            with self.assertRaises(TypeError):
                executor.fit_inner_estimator(
                    factory,
                    train,
                    validation,
                    basket=1,
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    eval_set=[frame],
                )
            self.assertEqual(factory.calls, 0)

    def test_internal_estimator_eval_set_is_exact_same_contract_validation(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="safe_eval", seed=0, inner_basket=1
            )
            train = contract.inner_training_batch(frame, 1)
            validation = contract.inner_validation_batch(frame, 1)
            recorder = self._recorder(contract, train, validation, run, audit)
            model = executor.fit_inner_estimator(
                SpyEstimator,
                train,
                validation,
                basket=1,
                feature_columns=["feature"],
                target_column="target",
                run_context=run,
                recorder=recorder,
                use_internal_eval_set=True,
            )
            self.assertEqual(set(model.fit_ids), {"T2", "T3", "T4"})
            self.assertEqual(set(model.eval_ids), {"T1"})
            self.assertTrue(set(model.eval_ids).isdisjoint(contract.outer_test_ids))

    def test_inner_preprocessor_is_fit_only_on_inner_train_and_applied_to_validation(self):
        contract, base = tiny_contract(), tiny_frame()
        mutated = base.copy()
        mutated.loc[mutated.curated_id.isin(["T1", "X1"]), ["feature", "feature_2"]] = [
            [1e12, -1e12],
            [-9e12, 8e12],
        ]
        processors = []
        for frame in (base, mutated):
            processor = SplitSafePreprocessor(contract, FitAuditTrail())
            processor.fit(
                contract.inner_training_batch(frame, 1),
                ["feature", "feature_2"],
                target_column="target",
            )
            processors.append(processor)
        self.assertEqual(processors[0].medians_, processors[1].medians_)
        self.assertEqual(processors[0].means_, processors[1].means_)
        self.assertEqual(processors[0].scales_, processors[1].scales_)
        self.assertEqual(processors[0].statistics_sha256_, processors[1].statistics_sha256_)

        audit = FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="inner_preprocess", seed=0, inner_basket=1
            )
            train = contract.inner_training_batch(base, 1)
            validation = contract.inner_validation_batch(base, 1)
            recorder = self._recorder(
                contract,
                train,
                validation,
                run,
                audit,
                feature_columns=("feature", "feature_2"),
                transform_sha256=processors[0].transform_sha256_,
            )
            model = executor.fit_inner_estimator(
                SpyEstimator,
                train,
                validation,
                basket=1,
                feature_columns=["feature", "feature_2"],
                target_column="target",
                run_context=run,
                recorder=recorder,
                preprocessor=processors[0],
                use_internal_eval_set=True,
            )
            self.assertEqual(set(model.eval_ids), {"T1"})

            outer_processor = SplitSafePreprocessor(contract, audit).fit(
                contract.outer_training_batch(base),
                ["feature", "feature_2"],
                target_column="target",
            )
            bad_run = contract.mint_run_context(
                Path(temporary), config_id="wrong_preprocess", seed=0, inner_basket=1
            )
            factory = CountingFactory()
            with self.assertRaises(SplitViolation):
                executor.fit_inner_estimator(
                    factory,
                    contract.inner_training_batch(base, 1),
                    contract.inner_validation_batch(base, 1),
                    basket=1,
                    feature_columns=["feature", "feature_2"],
                    target_column="target",
                    run_context=bad_run,
                    preprocessor=outer_processor,
                )
            self.assertEqual(factory.calls, 0)

    def test_factory_closure_capture_is_rejected_before_factory_invocation(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        outer_payload = frame.loc[frame.curated_id == "X1"].copy()

        def captured_factory():
            _ = outer_payload
            return SpyEstimator()

        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="closure", seed=0, inner_basket=None
            )
            with self.assertRaises(SplitViolation):
                executor.fit_outer_estimator(
                    captured_factory,
                    contract.outer_training_batch(frame),
                    ["feature"],
                    "target",
                    run_context=run,
                )

    def test_callable_defaults_partials_bound_state_and_callable_objects_are_scanned(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        outer_payload = frame.loc[frame.curated_id == "X1"].copy()

        def positional_default(payload=outer_payload):
            return SpyEstimator()

        def keyword_default(*, payload=outer_payload):
            return SpyEstimator()

        def partial_target(payload=None):
            return SpyEstimator()

        class BoundFactory:
            def __init__(self, payload):
                self.payload = payload

            def make(self):
                return SpyEstimator()

        class CallableFactory:
            model_config_sha256 = ONE_HASH

            def __call__(self, payload=outer_payload):
                return SpyEstimator()

        partial_with_state = functools.partial(partial_target)
        partial_with_state.hidden_payload = outer_payload
        factories = (
            positional_default,
            keyword_default,
            functools.partial(partial_target, outer_payload),
            functools.partial(partial_target, payload=outer_payload),
            partial_with_state,
            BoundFactory(outer_payload).make,
            CallableFactory(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, factory in enumerate(factories):
                with self.subTest(factory=index):
                    run = contract.mint_run_context(
                        Path(temporary),
                        config_id=f"callable_scan_{index}",
                        seed=0,
                        inner_basket=None,
                    )
                    with self.assertRaises(SplitViolation):
                        executor.fit_outer_estimator(
                            factory,
                            contract.outer_training_batch(frame),
                            ["feature"],
                            "target",
                            run_context=run,
                        )

            safe_partial = functools.partial(SpyEstimator)
            safe_partial.model_config_sha256 = ONE_HASH
            safe_run = contract.mint_run_context(
                Path(temporary), config_id="safe_partial", seed=0, inner_basket=None
            )
            model = executor.fit_outer_estimator(
                safe_partial,
                contract.outer_training_batch(frame),
                ["feature"],
                "target",
                run_context=safe_run,
            )
            self.assertEqual(model.fit_calls, 1)

    def test_inherited_init_default_is_rejected_before_exact_qa_factory_invocation(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        outer_payload = frame.loc[frame.curated_id == "X1"].copy()

        class CountingMeta(type):
            calls = 0

            def __call__(cls, *args, **kwargs):
                CountingMeta.calls += 1
                return super().__call__(*args, **kwargs)

        class Base(SpyEstimator, metaclass=CountingMeta):
            def __init__(self, payload=outer_payload):
                super().__init__()

        class Child(Base):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="inherited_init_qa", seed=0, inner_basket=None
            )
            with self.assertRaises(SplitViolation):
                executor.fit_outer_estimator(
                    Child,
                    contract.outer_training_batch(frame),
                    ["feature"],
                    "target",
                    run_context=run,
                )
        self.assertEqual(CountingMeta.calls, 0)

    def test_inherited_new_default_is_rejected_before_factory_invocation(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        outer_payload = frame.loc[frame.curated_id == "X1"].copy()

        class CountingMeta(type):
            calls = 0

            def __call__(cls, *args, **kwargs):
                CountingMeta.calls += 1
                return super().__call__(*args, **kwargs)

        class Base(SpyEstimator, metaclass=CountingMeta):
            def __new__(cls, payload=outer_payload):
                return super().__new__(cls)

        class Child(Base):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="inherited_new_qa", seed=0, inner_basket=None
            )
            with self.assertRaises(SplitViolation):
                executor.fit_outer_estimator(
                    Child,
                    contract.outer_training_batch(frame),
                    ["feature"],
                    "target",
                    run_context=run,
                )
        self.assertEqual(CountingMeta.calls, 0)

    def test_effective_multiple_inheritance_override_and_safe_defaults(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        outer_payload = frame.loc[frame.curated_id == "X1"].copy()

        class Bridge:
            pass

        class UnsafeInherited(SpyEstimator):
            def __init__(self, *, payload=outer_payload):
                super().__init__()

        class MultipleInherited(Bridge, UnsafeInherited):
            pass

        class SafeOverride(MultipleInherited):
            def __init__(self, *, payload=1):
                SpyEstimator.__init__(self)

        class SafeBase(SpyEstimator):
            def __init__(self, payload=1, *, enabled=True):
                SpyEstimator.__init__(self)

        class SafeInherited(SafeBase):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            unsafe_run = contract.mint_run_context(
                Path(temporary), config_id="multiple_inherited", seed=0, inner_basket=None
            )
            with self.assertRaises(SplitViolation):
                executor.fit_outer_estimator(
                    MultipleInherited,
                    contract.outer_training_batch(frame),
                    ["feature"],
                    "target",
                    run_context=unsafe_run,
                )
            for index, factory in enumerate((SafeOverride, SafeInherited)):
                run = contract.mint_run_context(
                    Path(temporary),
                    config_id=f"safe_inherited_{index}",
                    seed=0,
                    inner_basket=None,
                )
                model = executor.fit_outer_estimator(
                    factory,
                    contract.outer_training_batch(frame),
                    ["feature"],
                    "target",
                    run_context=run,
                )
                self.assertEqual(model.fit_calls, 1)

    def test_estimator_class_payload_is_rejected_before_fit(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        ClassPayloadEstimator.outer_payload = frame.loc[frame.curated_id == "X1"].copy()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                run = contract.mint_run_context(
                    Path(temporary), config_id="class_payload", seed=0, inner_basket=None
                )
                with self.assertRaises(SplitViolation):
                    executor.fit_outer_estimator(
                        ClassPayloadEstimator,
                        contract.outer_training_batch(frame),
                        ["feature"],
                        "target",
                        run_context=run,
                    )
        finally:
            ClassPayloadEstimator.outer_payload = None

    def test_frame_validation_attack_and_parent_dataset_reachability_rejected(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        train = contract.inner_training_batch(frame, 1)
        validation = contract.inner_validation_batch(frame, 1)
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="frame_attack", seed=0, inner_basket=1
            )
            recorder = self._recorder(contract, train, validation, run, audit)
            adapter = SpyFrameAdapter()
            with self.assertRaises(SplitViolation):
                executor.fit_inner_frame(
                    adapter,
                    train,
                    validation,
                    basket=1,
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    recorder=recorder,
                    maximum_epochs=2,
                    options={"validation_frame": frame.loc[frame.curated_id == "X1"]},
                )
            self.assertEqual(adapter.create_calls, 0)
            for payload in (
                frame.loc[frame.curated_id == "X1"].copy(),
                ParentSubset(frame.copy()),
                lambda: frame,
            ):
                poisoned = PayloadFrameAdapter(payload)
                with self.assertRaises(SplitViolation):
                    executor.fit_inner_frame(
                        poisoned,
                        train,
                        validation,
                        basket=1,
                        feature_columns=["feature"],
                        target_column="target",
                        run_context=run,
                        recorder=recorder,
                        maximum_epochs=2,
                    )
                self.assertEqual(poisoned.create_calls, 0)

    def test_outcome_and_group_predictors_rejected_at_all_boundaries(self):
        aliases = [
            "target",
            "permeability",
            "log_Papp",
            "replicate_spread",
            "curated_id",
            "source",
        ]
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        outer = contract.outer_training_batch(frame)
        outer_frame = contract.outer_frame_training_batch(frame)
        executor = SplitSafeFitExecutor(contract, audit)
        with tempfile.TemporaryDirectory() as temporary:
            for index, alias in enumerate(aliases):
                with self.subTest(alias=alias):
                    with self.assertRaises(SplitViolation):
                        SplitSafePreprocessor(contract, audit).fit(
                            outer, [alias], target_column="target"
                        )
                    factory = CountingFactory()
                    run = contract.mint_run_context(
                        Path(temporary),
                        config_id=f"estimator_{index}",
                        seed=0,
                        inner_basket=None,
                    )
                    with self.assertRaises(SplitViolation):
                        executor.fit_outer_estimator(
                            factory,
                            outer,
                            [alias],
                            "target",
                            run_context=run,
                        )
                    self.assertEqual(factory.calls, 0)
                    adapter = SpyFrameAdapter()
                    frame_run = contract.mint_run_context(
                        Path(temporary),
                        config_id=f"frame_{index}",
                        seed=0,
                        inner_basket=None,
                    )
                    with self.assertRaises(SplitViolation):
                        executor.fit_outer_frame(
                            adapter,
                            outer_frame,
                            feature_columns=[alias],
                            target_column="target",
                            run_context=frame_run,
                            fixed_epoch=3,
                        )
                    self.assertEqual(adapter.create_calls, 0)

    def test_frame_adapter_receives_only_projected_safe_columns(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="outer_dmpnn", seed=0, inner_basket=None
            )
            adapter = SpyFrameAdapter()
            executor.fit_outer_frame(
                adapter,
                contract.outer_frame_training_batch(frame),
                feature_columns=["feature", "feature_2"],
                target_column="target",
                run_context=run,
                fixed_epoch=17,
                options=SafeFitOptions(verbose=False),
            )
            self.assertEqual(adapter.outer_calls, 1)
            self.assertEqual(set(adapter.last_train_ids), contract.outer_train_ids)
            self.assertEqual(
                adapter.last_outer_columns,
                ("curated_id", "feature", "feature_2", "target"),
            )
            self.assertEqual(adapter.last_fixed_epoch, 17)

    def test_outer_frame_prediction_is_exact_bound_audited_and_single_use(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="outer_dmpnn_prediction", seed=3, inner_basket=None
            )
            adapter = SpyFrameAdapter()
            handle = executor.fit_outer_frame(
                adapter,
                contract.outer_frame_training_batch(frame),
                feature_columns=["feature", "feature_2"],
                target_column="target",
                run_context=run,
                fixed_epoch=17,
            )
            registry = executor._SplitSafeFitExecutor__sealed_outer_frame_states
            sealed_run = registry[handle.handle_token].run_context
            with self.assertRaises(AttributeError):
                object.__setattr__(sealed_run, "seed", 999)
            # Mutating the caller's original frame cannot change the executor's
            # internally sealed outer predictor values.
            frame.loc[frame["curated_id"] == "X1", "feature"] = -777.0
            result = executor.predict_outer_frame(handle)
            self.assertEqual(result.ids, ("X1",))
            self.assertEqual(result.predictions, (999.0,))
            self.assertEqual(result.ids_sha256, canonical_id_hash({"X1"}))
            self.assertEqual(len(result.prediction_sha256), 64)
            self.assertEqual(len(result.outer_fit_identity_sha256), 64)
            telemetry = dict(result.adapter_telemetry)
            self.assertEqual(adapter.outer_predict_calls, 0)
            self.assertEqual(telemetry["outer_predict_calls"], 1)
            self.assertNotEqual(telemetry["run_object_id"], id(sealed_run))
            prediction_audit = audit.records[-1]
            self.assertEqual(prediction_audit["operation"], "guarded_outer_frame_prediction")
            self.assertEqual(
                prediction_audit["outer_fit_identity_sha256"],
                result.outer_fit_identity_sha256,
            )
            self.assertEqual(
                prediction_audit["outer_fit_identity"]["outer_train_ids_sha256"],
                canonical_id_hash(contract.outer_train_ids),
            )
            self.assertEqual(
                prediction_audit["outer_fit_identity"]["outer_test_ids_sha256"],
                canonical_id_hash(contract.outer_test_ids),
            )
            self.assertEqual(
                prediction_audit["outer_fit_identity"]["feature_columns"],
                ("feature", "feature_2"),
            )
            self.assertEqual(
                len(prediction_audit["outer_fit_identity"]["outer_test_frame_sha256"]), 64
            )
            with self.assertRaises(SplitViolation):
                executor.predict_outer_frame(handle)
            self.assertEqual(adapter.outer_predict_calls, 0)

    def test_outer_frame_prediction_identity_mismatches_reject_before_adapter(self):
        cases = ("token", "value", "dtype", "order", "adapter", "model", "checkpoint")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
                executor = SplitSafeFitExecutor(contract, audit)
                run = contract.mint_run_context(
                    Path(temporary), config_id="outer_identity", seed=0, inner_basket=None
                )
                adapter = SpyFrameAdapter()
                handle = executor.fit_outer_frame(
                    adapter,
                    contract.outer_frame_training_batch(frame),
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    fixed_epoch=2,
                )
                self.assertFalse(hasattr(handle, "model"))
                attempted_handle = handle
                registry = executor._SplitSafeFitExecutor__sealed_outer_frame_states
                sealed_frame = registry[handle.handle_token].outer_prediction_frame
                if case == "token":
                    attempted_handle = replace(handle, handle_token="f" * 64)
                elif case == "value":
                    sealed_frame.loc[sealed_frame["curated_id"] == "X1", "feature"] = -777.0
                elif case == "dtype":
                    sealed_frame["feature"] = sealed_frame["feature"].astype("int64")
                elif case == "order":
                    sealed_frame.columns = tuple(reversed(sealed_frame.columns))
                elif case == "adapter":
                    adapter.best_epoch = 99
                elif case == "model":
                    registry[handle.handle_token].state.model.offset = 321.0
                elif case == "checkpoint":
                    (Path(run.checkpoint_dir) / "tampered.ckpt").write_text(
                        "tamper", encoding="utf-8"
                    )
                with self.assertRaises(SplitViolation):
                    executor.predict_outer_frame(attempted_handle)
                self.assertEqual(adapter.outer_predict_calls, 0)

    def test_outer_frame_prediction_rejects_order_payload_and_bad_outputs(self):
        class BadOutputAdapter(SpyFrameAdapter):
            def __init__(self, mode):
                super().__init__()
                self.mode = mode

            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                ids = tuple(outer_test_frame["curated_id"].astype(str))
                if self.mode == "array":
                    return FramePredictionOutput(np.zeros(len(ids), dtype=float))
                values = np.zeros(len(ids), dtype=float)
                if self.mode == "nonfinite":
                    values[0] = np.nan
                index = pd.Index(ids, name="curated_id")
                if self.mode == "reverse":
                    index = pd.Index(tuple(reversed(ids)), name="curated_id")
                return FramePredictionOutput(pd.Series(values, index=index))

        def two_test_case():
            contract = OuterFoldContract(
                1,
                {"T1", "T2", "T3", "T4"},
                {"X1", "X2"},
                {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
            )
            frame = pd.DataFrame(
                {
                    "curated_id": ["T1", "T2", "T3", "T4", "X1", "X2"],
                    "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "target": [1.0, 2.0, 3.0, 4.0, 99.0, 100.0],
                    "source": ["a", "b", "c", "d", "x", "y"],
                }
            )
            return contract, frame

        with tempfile.TemporaryDirectory() as temporary:
            contract, frame = two_test_case()
            audit, adapter = FitAuditTrail(), SpyFrameAdapter()
            executor = SplitSafeFitExecutor(contract, audit)
            run = contract.mint_run_context(
                Path(temporary), config_id="wrong_order", seed=0, inner_basket=None
            )
            handle = executor.fit_outer_frame(
                adapter,
                contract.outer_frame_training_batch(frame),
                feature_columns=["feature"],
                target_column="target",
                run_context=run,
                fixed_epoch=2,
            )
            frame = frame.iloc[::-1].reset_index(drop=True)
            result = executor.predict_outer_frame(handle)
            self.assertEqual(result.ids, ("X1", "X2"))
            self.assertEqual(result.predictions, (5.0, 6.0))
            self.assertEqual(adapter.outer_predict_calls, 0)
            self.assertEqual(dict(result.adapter_telemetry)["outer_predict_calls"], 1)

        with tempfile.TemporaryDirectory() as temporary:
            contract, frame = two_test_case()
            audit, adapter = FitAuditTrail(), SpyFrameAdapter()
            executor = SplitSafeFitExecutor(contract, audit)
            run = contract.mint_run_context(
                Path(temporary), config_id="prediction_payload", seed=0, inner_basket=None
            )
            handle = executor.fit_outer_frame(
                adapter,
                contract.outer_frame_training_batch(frame),
                feature_columns=["feature"],
                target_column="target",
                run_context=run,
                fixed_epoch=2,
            )
            adapter.payload = contract.outer_test_batch(frame).frame.copy()
            with self.assertRaises(SplitViolation):
                executor.predict_outer_frame(handle)
            self.assertEqual(adapter.outer_predict_calls, 0)

        for mode in ("array", "nonfinite", "reverse"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                contract, frame = two_test_case()
                audit, adapter = FitAuditTrail(), BadOutputAdapter(mode)
                executor = SplitSafeFitExecutor(contract, audit)
                run = contract.mint_run_context(
                    Path(temporary), config_id=f"bad_{mode}", seed=0, inner_basket=None
                )
                handle = executor.fit_outer_frame(
                    adapter,
                    contract.outer_frame_training_batch(frame),
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    fixed_epoch=2,
                )
                with self.assertRaises(SplitViolation):
                    executor.predict_outer_frame(handle)
                self.assertEqual(adapter.outer_predict_calls, 0)
                self.assertEqual(
                    audit.records[-1]["operation"],
                    "guarded_outer_frame_prediction_failure",
                )

    def test_outer_frame_seal_detects_parameter_buffer_and_predict_time_mutation(self):
        class StateDictAdapter(SpyFrameAdapter):
            def __init__(self, mutation=None):
                super().__init__()
                self.mutation = mutation

            def create_fresh_state(self, *, run_context):
                self.create_calls += 1
                return FreshTrainingState(StateDictBox(), StateBox(), StateBox())

            def predict_outer(self, *, state, outer_test_frame, feature_columns, run_context):
                ids = tuple(outer_test_frame["curated_id"].astype(str))
                result = pd.Series(
                    outer_test_frame[feature_columns[0]].to_numpy(float) + state.model.offset,
                    index=pd.Index(ids, name="curated_id"),
                    name="prediction",
                )
                if self.mutation == "model":
                    state.model.parameter[0] = 88.0
                elif self.mutation == "frame":
                    outer_test_frame.loc[0, feature_columns[0]] = -88.0
                return FramePredictionOutput(result, {"outer_predict_calls": 1})

        for changed_member in ("parameter", "buffer"):
            with self.subTest(changed_member=changed_member), tempfile.TemporaryDirectory() as temporary:
                contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
                executor, adapter = SplitSafeFitExecutor(contract, audit), StateDictAdapter()
                run = contract.mint_run_context(
                    Path(temporary), config_id=f"sealed_{changed_member}", seed=0, inner_basket=None
                )
                handle = executor.fit_outer_frame(
                    adapter,
                    contract.outer_frame_training_batch(frame),
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    fixed_epoch=2,
                )
                registry = executor._SplitSafeFitExecutor__sealed_outer_frame_states
                model = registry[handle.handle_token].state.model
                getattr(model, changed_member)[0] += 1.0
                with self.assertRaises(SplitViolation):
                    executor.predict_outer_frame(handle)
                self.assertEqual(adapter.outer_predict_calls, 0)

        for mutation in ("model", "frame"):
            with self.subTest(predict_mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
                executor, adapter = SplitSafeFitExecutor(contract, audit), StateDictAdapter(mutation)
                run = contract.mint_run_context(
                    Path(temporary), config_id=f"predict_{mutation}", seed=0, inner_basket=None
                )
                handle = executor.fit_outer_frame(
                    adapter,
                    contract.outer_frame_training_batch(frame),
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    fixed_epoch=2,
                )
                audit_count = len(audit.records)
                with self.assertRaises(SplitViolation):
                    executor.predict_outer_frame(handle)
                self.assertEqual(adapter.outer_predict_calls, 0)
                self.assertEqual(len(audit.records), audit_count + 1)
                self.assertEqual(
                    audit.records[-1]["operation"],
                    "guarded_outer_frame_prediction_failure",
                )

    def test_outer_frame_seals_adapter_predictor_run_and_failure_cleanup(self):
        class MutatingAdapter(SpyFrameAdapter):
            def __init__(self, mode):
                super().__init__()
                self.mode = mode
                self.runtime_state = {"nested": {"threshold": 1}}

            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                if self.mode == "model_config":
                    self.model_config_sha256 = TWO_HASH
                    self.critical_runtime_state = "changed"
                elif self.mode == "runtime":
                    self.runtime_state["nested"]["threshold"] = 999
                elif self.mode == "run_seed":
                    object.__setattr__(run_context, "seed", 999)
                elif self.mode == "predictor_attribute":
                    type(self).predict_outer.runtime_flag = "changed"
                elif self.mode == "exception":
                    raise RuntimeError("expected prediction exception")
                elif self.mode == "exception_after_runtime_mutation":
                    self.runtime_state["nested"]["threshold"] = 999
                    raise RuntimeError("expected exception after mutation")
                return valid_frame_prediction_output(
                    state, outer_test_frame, feature_columns
                )

        class DefaultMutationAdapter(SpyFrameAdapter):
            def predict_outer(
                self,
                *,
                state,
                outer_test_frame,
                feature_columns,
                run_context,
                _runtime={"threshold": 1},
            ):
                _runtime["threshold"] = 999
                return valid_frame_prediction_output(
                    state, outer_test_frame, feature_columns
                )

        closure_runtime = {"threshold": 1}

        class ClosureMutationAdapter(SpyFrameAdapter):
            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                closure_runtime["threshold"] = 999
                return valid_frame_prediction_output(
                    state, outer_test_frame, feature_columns
                )

        class InheritedRuntimeBase(SpyFrameAdapter):
            inherited_runtime = {"nested": {"threshold": 1}}

        class InheritedRuntimeMutationAdapter(InheritedRuntimeBase):
            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                self.inherited_runtime["nested"]["threshold"] = 999
                return valid_frame_prediction_output(
                    state, outer_test_frame, feature_columns
                )

        class HelperDefaultMutationAdapter(SpyFrameAdapter):
            def mutate_helper(self, runtime={"nested": {"threshold": 1}}):
                runtime["nested"]["threshold"] = 999

            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                self.mutate_helper()
                return valid_frame_prediction_output(
                    state, outer_test_frame, feature_columns
                )

        class SlotRuntimeMutationAdapter(SpyFrameAdapter):
            __slots__ = ("slot_runtime",)

            def __init__(self):
                super().__init__()
                self.slot_runtime = {"nested": {"threshold": 1}}

            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                self.slot_runtime["nested"]["threshold"] = 999
                return valid_frame_prediction_output(
                    state, outer_test_frame, feature_columns
                )

        cases = {
            "model_config": lambda: MutatingAdapter("model_config"),
            "nested_runtime": lambda: MutatingAdapter("runtime"),
            "run_seed": lambda: MutatingAdapter("run_seed"),
            "predictor_attribute": lambda: MutatingAdapter("predictor_attribute"),
            "predictor_default": DefaultMutationAdapter,
            "predictor_closure": ClosureMutationAdapter,
            "inherited_nested_runtime": InheritedRuntimeMutationAdapter,
            "helper_nested_default": HelperDefaultMutationAdapter,
            "slot_nested_runtime": SlotRuntimeMutationAdapter,
            "exception_cleanup": lambda: MutatingAdapter("exception"),
            "exception_after_runtime_mutation": lambda: MutatingAdapter(
                "exception_after_runtime_mutation"
            ),
        }
        for case, adapter_factory in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
                executor, adapter = SplitSafeFitExecutor(contract, audit), adapter_factory()
                run = contract.mint_run_context(
                    Path(temporary), config_id=f"repair7_{case}", seed=0, inner_basket=None
                )
                handle = executor.fit_outer_frame(
                    adapter,
                    contract.outer_frame_training_batch(frame),
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    fixed_epoch=2,
                )
                registry = executor._SplitSafeFitExecutor__sealed_outer_frame_states
                sealed_run = registry[handle.handle_token].run_context
                records_before = len(audit.records)
                with self.assertRaises(SplitViolation):
                    executor.predict_outer_frame(handle)
                self.assertEqual(run.seed, 0)
                self.assertEqual(sealed_run.seed, 0)
                self.assertEqual(len(registry), 0)
                self.assertEqual(len(audit.records), records_before + 1)
                self.assertEqual(
                    audit.records[-1]["operation"],
                    "guarded_outer_frame_prediction_failure",
                )
                self.assertTrue(audit.records[-1]["handle_consumed"])
                failure_count = len(audit.records)
                with self.assertRaises(SplitViolation):
                    executor.predict_outer_frame(handle)
                self.assertEqual(len(audit.records), failure_count)

    def test_outer_frame_rejects_nested_telemetry_and_noncanonical_output(self):
        class NestedTelemetryAdapter(SpyFrameAdapter):
            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                return valid_frame_prediction_output(
                    state,
                    outer_test_frame,
                    feature_columns,
                    telemetry={"nested_runtime": {"threshold": 1}},
                )

        class SubclassOutputAdapter(SpyFrameAdapter):
            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                ids = tuple(outer_test_frame["curated_id"].astype(str))
                predictions = PredictionSeriesSubclass(
                    outer_test_frame[feature_columns[0]].to_numpy(float),
                    index=pd.Index(ids, name="curated_id"),
                )
                return FramePredictionOutput(predictions, {"outer_predict_calls": 1})

        class OutputProcessingMutationAdapter(SpyFrameAdapter):
            def __init__(self):
                super().__init__()
                self.runtime_state = {"nested": {"threshold": 1}}

            def predict_outer(
                self, *, state, outer_test_frame, feature_columns, run_context
            ):
                import pandas as pandas_runtime

                original_to_numeric = pandas_runtime.to_numeric

                def mutate_during_output_processing(*args, **kwargs):
                    self.runtime_state["nested"]["threshold"] = 999
                    pandas_runtime.to_numeric = original_to_numeric
                    return original_to_numeric(*args, **kwargs)

                pandas_runtime.to_numeric = mutate_during_output_processing
                return valid_frame_prediction_output(
                    state, outer_test_frame, feature_columns
                )

        for case, adapter_factory in {
            "nested_telemetry": NestedTelemetryAdapter,
            "series_subclass": SubclassOutputAdapter,
            "output_processing_mutation": OutputProcessingMutationAdapter,
        }.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
                executor = SplitSafeFitExecutor(contract, audit)
                adapter = adapter_factory()
                run = contract.mint_run_context(
                    Path(temporary), config_id=case, seed=0, inner_basket=None
                )
                handle = executor.fit_outer_frame(
                    adapter,
                    contract.outer_frame_training_batch(frame),
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    fixed_epoch=2,
                )
                records_before = len(audit.records)
                with self.assertRaises(SplitViolation):
                    executor.predict_outer_frame(handle)
                self.assertEqual(len(audit.records), records_before + 1)
                self.assertEqual(
                    audit.records[-1]["operation"],
                    "guarded_outer_frame_prediction_failure",
                )
                if case == "output_processing_mutation":
                    self.assertEqual(
                        adapter.runtime_state["nested"]["threshold"], 999
                    )
                with self.assertRaises(SplitViolation):
                    executor.predict_outer_frame(handle)
                self.assertEqual(len(audit.records), records_before + 1)

    def test_outer_frame_telemetry_is_returned_frozen_not_stored_on_adapter(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor, adapter = SplitSafeFitExecutor(contract, audit), SpyFrameAdapter()
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="repair7_telemetry", seed=0, inner_basket=None
            )
            handle = executor.fit_outer_frame(
                adapter,
                contract.outer_frame_training_batch(frame),
                feature_columns=["feature"],
                target_column="target",
                run_context=run,
                fixed_epoch=2,
            )
            result = executor.predict_outer_frame(handle)
        telemetry = dict(result.adapter_telemetry)
        self.assertEqual(telemetry["outer_predict_calls"], 1)
        self.assertEqual(adapter.outer_predict_calls, 0)
        self.assertEqual(adapter.last_outer_predict_ids, ())
        with self.assertRaises(TypeError):
            result.adapter_telemetry[0] = ("changed", True)
        self.assertEqual(
            audit.records[-1]["adapter_telemetry"], telemetry
        )

    def test_inner_frame_identity_mismatches_fail_before_adapter_invocation(self):
        mismatch_cases = ("schema", "target", "transform", "model", "run", "basket")
        for case in mismatch_cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
                frame = frame.assign(target_2=frame["target"] + 0.5)
                executor = SplitSafeFitExecutor(contract, audit)
                train = contract.inner_training_batch(frame, 1)
                validation = contract.inner_validation_batch(frame, 1)
                run = contract.mint_run_context(
                    Path(temporary), config_id="identity", seed=0, inner_basket=1
                )
                recorder_features = ("feature", "feature_2") if case == "schema" else ("feature",)
                recorder = self._recorder(
                    contract,
                    train,
                    validation,
                    run,
                    audit,
                    feature_columns=recorder_features,
                )
                adapter = SpyFrameAdapter()
                fit_features = (
                    ("feature_2", "feature") if case == "schema" else ("feature",)
                )
                fit_target = "target_2" if case == "target" else "target"
                fit_train, fit_validation, fit_run, fit_basket = train, validation, run, 1
                if case == "transform":
                    adapter.transform_sha256 = TWO_HASH
                if case == "model":
                    adapter.model_config_sha256 = TWO_HASH
                if case == "run":
                    fit_run = contract.mint_run_context(
                        Path(temporary), config_id="identity_other_run", seed=0, inner_basket=1
                    )
                if case == "basket":
                    fit_basket = 2
                    fit_train = contract.inner_training_batch(frame, 2)
                    fit_validation = contract.inner_validation_batch(frame, 2)
                    fit_run = contract.mint_run_context(
                        Path(temporary), config_id="identity", seed=0, inner_basket=2
                    )
                with self.assertRaises(SplitViolation):
                    executor.fit_inner_frame(
                        adapter,
                        fit_train,
                        fit_validation,
                        basket=fit_basket,
                        feature_columns=fit_features,
                        target_column=fit_target,
                        run_context=fit_run,
                        recorder=recorder,
                        maximum_epochs=2,
                    )
                self.assertEqual(adapter.create_calls, 0)
                self.assertEqual(adapter.inner_calls, 0)

    def test_guarded_evaluator_mints_complete_prediction_bound_histories(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        histories = []
        with tempfile.TemporaryDirectory() as temporary:
            for basket, best_epoch in zip(range(1, 5), (2, 3, 4, 5)):
                train = contract.inner_training_batch(frame, basket)
                validation = contract.inner_validation_batch(frame, basket)
                run = contract.mint_run_context(
                    Path(temporary),
                    config_id="guarded_histories",
                    seed=0,
                    inner_basket=basket,
                )
                recorder = self._recorder(contract, train, validation, run, audit)
                adapter = SpyFrameAdapter(best_epoch=best_epoch)
                executor.fit_inner_frame(
                    adapter,
                    train,
                    validation,
                    basket=basket,
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    recorder=recorder,
                    maximum_epochs=6,
                )
                histories.append(recorder.finalize())
                self.assertEqual(set(adapter.last_train_ids), set(train.ids))
                self.assertEqual(set(adapter.last_validation_ids), set(validation.ids))
                self.assertTrue(set(adapter.last_validation_ids).isdisjoint(contract.outer_test_ids))
                for event in histories[-1].events:
                    self.assertEqual(event.validation_ids_sha256, canonical_id_hash(validation.ids))
                    self.assertEqual(len(event.prediction_sha256), 64)
                    self.assertEqual(len(event.target_sha256), 64)
                    self.assertEqual(event.metric_identity, "mean_absolute_error")
                self.assertEqual(adapter.last_predict_columns, ("curated_id", "feature"))
            self.assertEqual(contract.select_stopping_epoch(histories, audit), 4)
            with self.assertRaises(SplitViolation):
                contract.select_stopping_epoch(histories, audit)

    def test_stopping_selector_rejects_each_cross_history_identity_mismatch(self):
        for mismatch in ("config", "model", "evaluator", "schema", "target", "transform"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as temporary:
                contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
                frame = frame.assign(target_2=frame["target"] + 0.5)
                executor = SplitSafeFitExecutor(contract, audit)
                histories = []
                for basket in range(1, 5):
                    is_changed = basket == 4
                    config_id = "stopping_identity"
                    if mismatch == "config" and is_changed:
                        config_id = "different_config"
                    run = contract.mint_run_context(
                        Path(temporary), config_id=config_id, seed=0, inner_basket=basket
                    )
                    feature_columns = (
                        ("feature_2",) if mismatch == "schema" and is_changed else ("feature",)
                    )
                    target_column = (
                        "target_2" if mismatch == "target" and is_changed else "target"
                    )
                    adapter = (
                        AlternateSpyFrameAdapter(best_epoch=2)
                        if mismatch == "evaluator" and is_changed
                        else SpyFrameAdapter(best_epoch=2)
                    )
                    if mismatch == "model" and is_changed:
                        adapter.model_config_sha256 = TWO_HASH
                    if mismatch == "transform" and is_changed:
                        adapter.transform_sha256 = TWO_HASH
                        adapter.fitted_transform_sha256 = TWO_HASH
                    train = contract.inner_training_batch(frame, basket)
                    validation = contract.inner_validation_batch(frame, basket)
                    recorder = self._recorder(
                        contract,
                        train,
                        validation,
                        run,
                        audit,
                        feature_columns=feature_columns,
                        target_column=target_column,
                        transform_sha256=adapter.transform_sha256,
                        model_config_sha256=adapter.model_config_sha256,
                    )
                    executor.fit_inner_frame(
                        adapter,
                        train,
                        validation,
                        basket=basket,
                        feature_columns=feature_columns,
                        target_column=target_column,
                        run_context=run,
                        recorder=recorder,
                        maximum_epochs=2,
                    )
                    histories.append(recorder.finalize())
                with self.assertRaises(SplitViolation):
                    contract.select_stopping_epoch(histories, audit)

    def test_opaque_loss_wrong_ids_missing_basket_and_outer_validation_rejected(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        with self.assertRaises(SplitViolation):
            contract.select_stopping_epoch([{"loss": [0.1]}] * 4, audit)
        with self.assertRaises(SplitViolation):
            contract.select_stopping_epoch([], audit)
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="outer_val", seed=0, inner_basket=1
            )
            with self.assertRaises(SplitViolation):
                contract.create_inner_evaluation_recorder(
                    contract.inner_training_batch(frame, 1),
                    contract.outer_test_batch(frame),
                    basket=1,
                    feature_columns=["feature"],
                    target_column="target",
                    metric_identity="mean_absolute_error",
                    run_context=run,
                    transform_sha256=ZERO_HASH,
                    model_config_sha256=ONE_HASH,
                    checkpoint_sha256=TWO_HASH,
                    audit=audit,
                )

    def test_altered_history_and_missing_basket_are_rejected(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        histories = []
        with tempfile.TemporaryDirectory() as temporary:
            for basket in range(1, 5):
                train = contract.inner_training_batch(frame, basket)
                validation = contract.inner_validation_batch(frame, basket)
                run = contract.mint_run_context(
                    Path(temporary), config_id="alter_histories", seed=0, inner_basket=basket
                )
                recorder = self._recorder(contract, train, validation, run, audit)
                adapter = SpyFrameAdapter(best_epoch=1)
                executor.fit_inner_frame(
                    adapter,
                    train,
                    validation,
                    basket=basket,
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=run,
                    recorder=recorder,
                    maximum_epochs=2,
                )
                histories.append(recorder.finalize())
            altered = list(histories)
            altered[0] = replace(altered[0], validation_ids=("X1",))
            with self.assertRaises(SplitViolation):
                contract.select_stopping_epoch(altered, audit)
            with self.assertRaises(SplitViolation):
                contract.select_stopping_epoch(histories[:3], audit)

    def test_checkpoint_namespace_restore_and_mutable_state_are_isolated(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        outer = contract.outer_frame_training_batch(frame)
        with tempfile.TemporaryDirectory() as temporary:
            poisoned = contract.mint_run_context(
                Path(temporary), config_id="poisoned", seed=0, inner_basket=None
            )
            (Path(poisoned.checkpoint_dir) / "latest.ckpt").write_text("poison", encoding="utf-8")
            adapter = SpyFrameAdapter()
            with self.assertRaises(SplitViolation):
                executor.fit_outer_frame(
                    adapter,
                    outer,
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=poisoned,
                    fixed_epoch=2,
                )
            self.assertEqual(adapter.create_calls, 0)

            restored = contract.mint_run_context(
                Path(temporary), config_id="restore", seed=0, inner_basket=None
            )
            with self.assertRaises(SplitViolation):
                executor.fit_outer_frame(
                    SpyFrameAdapter(),
                    outer,
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=restored._replace(restore=True),
                    fixed_epoch=2,
                )

            reuser = ReusingFrameAdapter()
            first = contract.mint_run_context(
                Path(temporary), config_id="fresh_a", seed=0, inner_basket=None
            )
            with self.assertRaises(SplitViolation):
                executor.fit_outer_frame(
                    reuser,
                    outer,
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=first,
                    fixed_epoch=2,
                )
            self.assertEqual(reuser.outer_calls, 1)
            second = contract.mint_run_context(
                Path(temporary), config_id="fresh_b", seed=0, inner_basket=None
            )
            calls_before = reuser.outer_calls
            with self.assertRaises(SplitViolation):
                executor.fit_outer_frame(
                    reuser,
                    outer,
                    feature_columns=["feature"],
                    target_column="target",
                    run_context=second,
                    fixed_epoch=2,
                )
            self.assertEqual(reuser.outer_calls, calls_before)

    def test_sample_weights_are_contract_issued_training_sidecars(self):
        contract, frame, audit = tiny_contract(), tiny_frame(), FitAuditTrail()
        executor = SplitSafeFitExecutor(contract, audit)
        outer = contract.outer_training_batch(frame)
        weights = contract.training_weights(outer, "weight")
        self.assertEqual(weights.ids_sha256, canonical_id_hash(outer.ids))
        with tempfile.TemporaryDirectory() as temporary:
            run = contract.mint_run_context(
                Path(temporary), config_id="weighted", seed=0, inner_basket=None
            )
            model = executor.fit_outer_estimator(
                SpyEstimator,
                outer,
                ["feature"],
                "target",
                run_context=run,
                sample_weight=weights,
            )
            self.assertEqual(model.fit_calls, 1)
        with self.assertRaises(SplitViolation):
            contract.training_weights(outer, "permeability")

    def test_machine_policy_declares_closed_boundary(self):
        policy = fit_boundary_policy()
        self.assertFalse(policy["generic_fit_kwargs"])
        self.assertFalse(policy["caller_supplied_loss_curves"])
        self.assertTrue(policy["parent_dataset_reachability_forbidden"])
        self.assertTrue(policy["fresh_model_optimizer_scheduler_per_run"])
        self.assertTrue(policy["outer_frame_prediction_outcome_free_projection"])
        self.assertTrue(policy["outer_frame_prediction_id_indexed"])
        self.assertTrue(policy["outer_frame_prediction_single_use"])
        self.assertFalse(policy["outer_frame_prediction_caller_supplied_frame"])
        self.assertFalse(policy["outer_frame_fit_returns_mutable_state"])
        self.assertTrue(policy["outer_frame_executor_private_registry"])
        self.assertTrue(policy["outer_frame_prediction_handle_only"])
        self.assertTrue(policy["outer_frame_run_context_tuple_immutable"])
        self.assertTrue(policy["outer_frame_run_context_disposable_adapter_copy"])
        self.assertTrue(policy["outer_frame_adapter_pre_post_deep_content_digest"])
        self.assertTrue(policy["outer_frame_adapter_mro_slots_helpers_content_digest"])
        self.assertTrue(policy["outer_frame_predictor_defaults_closures_attributes_digest"])
        self.assertTrue(policy["outer_frame_prediction_telemetry_detached"])
        self.assertTrue(policy["outer_frame_prediction_telemetry_scalar_only"])
        self.assertTrue(
            policy["outer_frame_prediction_final_post_output_state_verification"]
        )
        self.assertTrue(policy["outer_frame_prediction_failure_audited"])
        self.assertFalse(policy["caller_controlled_state_digest"])
        self.assertIn("predict_outer_frame", policy["typed_prediction_paths"])
        self.assertFalse(policy["r1_authorized"])

    def test_official_fit_signatures_have_no_var_keyword_escape_hatch(self):
        self.assertFalse(hasattr(SplitSafeFitExecutor, "fit_estimator"))
        self.assertFalse(hasattr(SplitSafeFitExecutor, "fit_frame"))
        for name in (
            "fit_outer_estimator",
            "fit_inner_estimator",
            "fit_outer_frame",
            "fit_inner_frame",
            "predict_outer_frame",
        ):
            parameters = inspect.signature(getattr(SplitSafeFitExecutor, name)).parameters
            self.assertNotIn(
                inspect.Parameter.VAR_KEYWORD,
                {parameter.kind for parameter in parameters.values()},
                name,
            )
        self.assertEqual(
            tuple(inspect.signature(SplitSafeFitExecutor.predict_outer_frame).parameters),
            ("self", "handle"),
        )

    def test_committed_machine_policy_matches_code(self):
        import json

        observed = json.loads(
            (ROOT / "artifacts" / "v2_r0" / "fit_boundary_policy.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(observed, fit_boundary_policy())


if __name__ == "__main__":
    unittest.main()
