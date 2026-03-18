"""Unit tests for configuration parsing and dataset YAML auto-configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml


class TestConfigParsing(unittest.TestCase):
    """Unit tests for config parsing and dataset YAML handling."""

    def test_parse_basic_yaml_config(self) -> None:
        """Test parsing a basic YAML config file."""
        config_data = {
            "model": {"num_classes": 4, "proto_k": 24},
            "train": {"batch_size": 8, "epochs": 100},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with open(config_path, "r") as handle:
                loaded = yaml.safe_load(handle)
            
            self.assertEqual(loaded["model"]["num_classes"], 4)
            self.assertEqual(loaded["train"]["batch_size"], 8)
        finally:
            Path(config_path).unlink()

    def test_dataset_yaml_class_names_list(self) -> None:
        """Test dataset YAML with class names as list."""
        data_yaml = {
            "train": "F:/data/train/images",
            "val": "F:/data/valid/images",
            "nc": 4,
            "names": ["ball", "goalkeeper", "player", "referee"],
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data_yaml, f)
            yaml_path = f.name
        
        try:
            with open(yaml_path, "r") as handle:
                loaded = yaml.safe_load(handle)
            
            self.assertEqual(loaded["nc"], 4)
            self.assertIsInstance(loaded["names"], list)
            self.assertEqual(len(loaded["names"]), 4)
            self.assertEqual(loaded["names"][0], "ball")
        finally:
            Path(yaml_path).unlink()

    def test_dataset_yaml_class_names_dict(self) -> None:
        """Test dataset YAML with class names as dict (Roboflow format)."""
        data_yaml = {
            "train": "F:/data/train/images",
            "val": "F:/data/valid/images",
            "nc": 4,
            "names": {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data_yaml, f)
            yaml_path = f.name
        
        try:
            with open(yaml_path, "r") as handle:
                loaded = yaml.safe_load(handle)
            
            self.assertEqual(loaded["nc"], 4)
            self.assertIsInstance(loaded["names"], dict)
            self.assertEqual(loaded["names"][0], "ball")
            
            # Convert dict to list
            names_list = [name for _, name in sorted(loaded["names"].items())]
            self.assertEqual(names_list, ["ball", "goalkeeper", "player", "referee"])
        finally:
            Path(yaml_path).unlink()

    def test_dataset_yaml_path_resolution(self) -> None:
        """Test that dataset paths are correctly specified."""
        data_yaml = {
            "train": "F:/data/train/images",
            "val": "F:/data/valid/images",
            "test": "F:/data/test/images",
            "nc": 2,
            "names": ["class0", "class1"],
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data_yaml, f)
            yaml_path = f.name
        
        try:
            with open(yaml_path, "r") as handle:
                loaded = yaml.safe_load(handle)
            
            self.assertIn("train", loaded)
            self.assertIn("val", loaded)
            self.assertIn("test", loaded)
            self.assertTrue(loaded["train"].endswith("images"))
        finally:
            Path(yaml_path).unlink()

    def test_config_with_nested_structure(self) -> None:
        """Test parsing config with nested structure."""
        config_data = {
            "model": {
                "backbone": {"type": "resnet", "depth": 50},
                "head": {"num_classes": 4, "proto_k": 24},
            },
            "train": {
                "optimizer": {"type": "adamw", "lr": 0.001},
                "scheduler": {"type": "cosine", "warmup_epochs": 3},
            },
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with open(config_path, "r") as handle:
                loaded = yaml.safe_load(handle)
            
            self.assertEqual(loaded["model"]["backbone"]["type"], "resnet")
            self.assertEqual(loaded["train"]["optimizer"]["lr"], 0.001)
        finally:
            Path(config_path).unlink()

    def test_config_with_defaults(self) -> None:
        """Test that missing keys can be handled with defaults."""
        config_data = {"model": {"num_classes": 4}}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with open(config_path, "r") as handle:
                loaded = yaml.safe_load(handle)
            
            # Test default value handling
            proto_k = loaded.get("model", {}).get("proto_k", 24)
            self.assertEqual(proto_k, 24)
        finally:
            Path(config_path).unlink()

    def test_empty_config(self) -> None:
        """Test handling of empty config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            config_path = f.name
        
        try:
            with open(config_path, "r") as handle:
                loaded = yaml.safe_load(handle)
            
            self.assertEqual(loaded, {})
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    unittest.main()
