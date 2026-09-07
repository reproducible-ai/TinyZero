from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = ROOT / ".gitignore"
WORKFLOW = ROOT / ".treqs" / "workflows" / "tinyzero-canary.yaml"
PACKAGER = ROOT / ".treqs" / "scripts" / "package_tinyzero_canary.py"
PREPROCESSOR = ROOT / "examples" / "data_preprocess" / "countdown.py"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


class TinyZeroCanaryContractTests(unittest.TestCase):
    def test_roar_runtime_state_cannot_dirty_the_training_checkout(self):
        gitignore = GITIGNORE.read_text().splitlines()

        self.assertIn(".roar/", gitignore)
        self.assertIn("models/", gitignore)
        self.assertIn("data/countdown-canary/", gitignore)
        self.assertIn("artifacts/", gitignore)

    def test_workflow_is_a_single_optimizer_step_l40s_canary(self):
        workflow = WORKFLOW.read_text()

        self.assertIn("Qwen/Qwen2.5-0.5B", workflow)
        self.assertIn("flash-attn==2.7.0.post2", workflow)
        self.assertIn("setuptools", workflow)
        self.assertIn("wheel", workflow)
        self.assertIn("'ray==2.10.0'", workflow)
        self.assertIn("'transformers==4.46.0'", workflow)
        self.assertIn(f"--revision {MODEL_REVISION}", workflow)
        self.assertIn("actor_rollout_ref.model.path=models/Qwen2.5-0.5B", workflow)
        self.assertIn("--canary_fixture", workflow)
        self.assertIn("algorithm.adv_estimator=grpo", workflow)
        self.assertIn("data.train_batch_size=8", workflow)
        self.assertIn("actor_rollout_ref.rollout.n=2", workflow)
        # Eight prompts times two rollouts form one 16-sample actor mini-batch.
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=16", workflow)
        self.assertIn("trainer.n_gpus_per_node=1", workflow)
        self.assertIn("trainer.total_training_steps=1", workflow)
        self.assertIn("trainer.save_freq=1", workflow)
        self.assertIn("WANDB_MODE=disabled", workflow)
        self.assertIn("roar config set ray.enabled false", workflow)
        self.assertIn("roar run -n train", workflow)
        self.assertIn("roar put artifacts/tinyzero-canary/model.safetensors", workflow)
        self.assertIn("--private --yes --no-tag", workflow)

    def test_dataset_preparation_pins_the_hugging_face_revision(self):
        source = PREPROCESSOR.read_text()

        self.assertIn("--dataset_revision", source)
        self.assertIn("revision=args.dataset_revision", source)
        self.assertIn("os.makedirs(local_dir, exist_ok=True)", source)
        self.assertIn("--canary_fixture", source)
        self.assertIn("build_canary_fixture", source)

    def test_packager_emits_harness_receipts_and_load_verifies_safetensors(self):
        source = PACKAGER.read_text()

        self.assertIn('"reproai.artifact/v1"', source)
        self.assertIn('"optimizerSteps": 1', source)
        self.assertIn("safe_open", source)
        self.assertIn('print("E2E_ARTIFACT="', source)
        self.assertIn('print("E2E_RESULT="', source)


if __name__ == "__main__":
    unittest.main()
