"""
Diagnostic script: inspect model.pkl to find out which azureml
classes it embeds, then attempt to load it with mock stubs.
Run with: conda run -n churnsight python inspect_model.py
"""

import sys
import types
import pickle

needed = []

class InspectingUnpickler(pickle.Unpickler):
    """Intercepts every class lookup during unpickling."""
    def find_class(self, module, name):
        if module.startswith('azureml'):
            entry = f"{module}.{name}"
            if entry not in needed:
                needed.append(entry)
                print(f"  NEEDS: {entry}")
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            # Build mock stubs on the fly so we can see how far we get
            parts = module.split('.')
            for i in range(len(parts)):
                parent = '.'.join(parts[:i+1])
                if parent not in sys.modules:
                    sys.modules[parent] = types.ModuleType(parent)
            mock_module = sys.modules[module]

            class MockClass:
                _mock_name = f"{module}.{name}"
                def __init__(self, *a, **kw): pass
                def __reduce__(self): return (MockClass, ())
                def predict(self, X): raise NotImplementedError(self._mock_name)
                def predict_proba(self, X): raise NotImplementedError(self._mock_name)

            MockClass.__name__ = name
            MockClass.__qualname__ = name
            setattr(mock_module, name, MockClass)
            return MockClass

print("=" * 60)
print("Inspecting model.pkl for azureml dependencies...")
print("=" * 60)

try:
    with open('model.pkl', 'rb') as f:
        model = InspectingUnpickler(f).load()

    print("\n✅ Model loaded with mocks!")
    print(f"   Type      : {type(model)}")
    print(f"   Has classes_: {hasattr(model, 'classes_')}")

    if hasattr(model, 'classes_'):
        print(f"   Classes   : {model.classes_}")

    # Try a minimal prediction to see if mocks survive inference
    import pandas as pd
    sample = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": False, "Partner": False,
        "Dependents": False, "tenure": 12, "PhoneService": True,
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": True,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.0, "TotalCharges": 800.0,
    }])

    try:
        pred  = model.predict(sample)
        proba = model.predict_proba(sample)
        print(f"\n🎉 INFERENCE WORKS WITH MOCKS!")
        print(f"   predict()       → {pred}")
        print(f"   predict_proba() → {proba}")
        print("\n💡 Conclusion: We do NOT need to install azureml.")
        print("   We can use a mock-based loader in app.py.\n")
    except NotImplementedError as e:
        print(f"\n⚠️  Mocks loaded the pickle but inference called real azureml code: {e}")
        print("   Conclusion: We DO need to install azureml packages.\n")
    except Exception as e:
        print(f"\n⚠️  Inference failed: {e}\n")

except Exception as e:
    print(f"\n❌ Could not load even with mocks: {e}\n")

print("=" * 60)
print(f"Total azureml dependencies found: {len(needed)}")
for n in needed:
    print(f"  - {n}")
print("=" * 60)
