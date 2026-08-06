import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SR_ROOT = ROOT / "SR"
sys.path.insert(0, str(SR_ROOT))

import audioseal_integration
from audioseal_integration import AudioSealWatermarker, checkpoint_lora_config
from datasets import SpeechCommandsDataset
from embed_trigger import poison_output, select_poison_files
from poison_manifest import propagate_manifest, read_manifest, validate_manifest, write_manifest
from split_speech_commands import deterministic_split, speech_commands_split
import models


def write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes((np.zeros(1600, dtype=np.int16)).tobytes())


class PipelineTests(unittest.TestCase):
    def test_first_party_models_run_on_cpu(self):
        inputs = torch.zeros(1, 1, 80, 87)
        self.assertEqual(tuple(models.create_model("resnet18", 10, 1)(inputs).shape), (1, 10))
        self.assertEqual(tuple(models.create_model("lstm", 10, 1)(inputs).shape), (1, 10))

    def test_feature_dataset_uses_stable_class_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "yes").mkdir()
            (root / "left").mkdir()
            np.save(root / "yes" / "a.npy", np.ones((1, 3, 4), dtype=np.float32))
            np.save(root / "left" / "b.npy", np.zeros((1, 3, 4), dtype=np.float32))
            dataset = SpeechCommandsDataset(root, num_classes=10)
            labels = {path.parent.name: label for path, label in dataset.samples}
            self.assertEqual(labels, {"yes": 0, "left": 4})
            self.assertEqual(tuple(dataset[0][0].shape), (1, 3, 4))

    def test_label_modes_map_output_class_and_filename(self):
        self.assertEqual(
            poison_output("yes", "speaker_nohash_0.wav", "left", "label-flip"),
            ("left", "yes_speaker_nohash_0.wav"),
        )
        self.assertEqual(
            poison_output("yes", "speaker_nohash_0.wav", "left", "clean-label"),
            ("yes", "speaker_nohash_0.wav"),
        )

    def test_split_and_poison_selection_are_exact_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for class_name in ("yes", "no", "left"):
                for index in range(4):
                    path = root / class_name / f"{index}.wav"
                    write_wav(path)
                    files.append(path)
            first = deterministic_split(files, test_ratio=0.25, validation_ratio=0.25, seed=7)
            second = deterministic_split(reversed(files), test_ratio=0.25, validation_ratio=0.25, seed=7)
            self.assertEqual(first, second)
            self.assertEqual([len(split) for split in first], [6, 3, 3])
            self.assertFalse(set(first[1]) & set(first[2]))
            selected, total = select_poison_files(root, ("yes", "no", "left"), "left", 0.25, 9)
            self.assertEqual((len(selected), total), (3, 12))
            self.assertTrue(all(path.parent.name != "left" for path in selected))
            self.assertEqual(selected, select_poison_files(root, ("yes", "no", "left"), "left", 0.25, 9)[0])

    def test_official_validation_and_test_lists_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(4):
                path = root / "yes" / f"speaker_{index}.wav"
                write_wav(path)
                paths.append(path)
            (root / "validation_list.txt").write_text("yes/speaker_1.wav\n", encoding="utf-8")
            (root / "testing_list.txt").write_text("yes/speaker_2.wav\n", encoding="utf-8")
            train, validation, test = speech_commands_split(root, ("yes",))
            self.assertEqual(validation, [paths[1]])
            self.assertEqual(test, [paths[2]])
            self.assertEqual(train, [paths[0], paths[3]])

    def test_poison_manifest_propagates_and_rejects_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wav"
            destination = Path(directory) / "features"
            settings = {
                "classes": ["yes", "left"],
                "target_label": "left",
                "label_mode": "clean-label",
            }
            write_manifest(source, settings)
            self.assertEqual(propagate_manifest(source, destination), settings)
            manifest = read_manifest(destination)
            validate_manifest(manifest, classes=("yes", "left"), target_label="left",
                              label_mode="clean-label")
            with self.assertRaisesRegex(ValueError, "target"):
                validate_manifest(manifest, classes=("yes", "left"), target_label="yes")

    def test_sr_modules_import_as_package_and_package_config_exists(self):
        import SR.audioseal_integration
        import SR.embed_trigger
        import SR.evaluate
        import SR.extract_features
        import SR.extract_poison_features
        import SR.extract_test_poison_features
        import SR.split_speech_commands
        import SR.train
        from SR.common import DEFAULT_CONFIG, load_config

        self.assertTrue(DEFAULT_CONFIG.is_file())
        self.assertIn("path", load_config())

    def test_checkpoint_metadata_and_explicit_output_scaling(self):
        checkpoint = {
            "args": {"lora_rank": 8, "lora_alpha": 16.0},
            "state_dict": {"decoder.x.lora_A.weight": torch.zeros(8, 2, 1)},
        }
        rank, alpha, _ = checkpoint_lora_config(checkpoint)
        self.assertEqual((rank, alpha), (8, 16.0))

        class Generator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def get_watermark(self, wave, sample_rate):
                return torch.full_like(wave, 0.1)

        watermarker = AudioSealWatermarker.__new__(AudioSealWatermarker)
        watermarker.device = torch.device("cpu")
        watermarker.sample_rate = 16000
        watermarker.watermark_scale = 2.0
        watermarker.generator = Generator()
        output, rate = watermarker.embed_watermark(np.zeros(8, dtype=np.float32), 16000)
        np.testing.assert_allclose(output, 0.2)
        self.assertEqual(rate, 16000)

    def test_constructor_uses_mode_specific_scale_defaults(self):
        class Generator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

        class AudioSeal:
            @staticmethod
            def load_generator(model_card):
                return Generator()

        checkpoint = {
            "args": {"lora_rank": 8, "lora_alpha": 16.0},
            "state_dict": {},
        }
        with patch.object(audioseal_integration, "import_audioseal", return_value=AudioSeal), \
                patch.object(audioseal_integration, "inject_lora", side_effect=lambda model, rank, alpha: model), \
                patch.object(audioseal_integration.torch, "load", return_value=checkpoint):
            base = AudioSealWatermarker(mode="base", device="cpu")
            fine_tuned = AudioSealWatermarker(mode="lora-ft", checkpoint_path="unused.pth", device="cpu")
            explicit = AudioSealWatermarker(mode="lora-ft", checkpoint_path="unused.pth",
                                             watermark_scale=2.5, device="cpu")
        self.assertEqual(base.watermark_scale, 5.0)
        self.assertEqual(fine_tuned.watermark_scale, 1.0)
        self.assertEqual(explicit.watermark_scale, 2.5)


if __name__ == "__main__":
    unittest.main()
