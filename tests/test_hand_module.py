import unittest
import sys
import os

# Add 'src' to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modules.Hand_Module.hand_engine import HandEngine

class TestHandModule(unittest.TestCase):
    def test_engine_initialization(self):
        """Check if HandEngine can load with the correct model path."""
        # Calculate the absolute path to the 'models' folder in the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(project_root, "models")
        
        try:
            # Pass the required 'model_dir' argument
            engine = HandEngine(model_dir=model_dir)
            self.assertIsNotNone(engine.detector)
            print("\n[Test] HandEngine initialized successfully with models.")
        except Exception as e:
            self.fail(f"HandEngine failed to initialize: {e}")

if __name__ == '__main__':
    unittest.main()