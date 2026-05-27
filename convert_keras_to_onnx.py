"""
Convierte el modelo CNN Keras (.keras) a ONNX (.onnx) para evitar cargar
TensorFlow en runtime. Solo necesitas ejecutar esto UNA VEZ.

Uso:
  conda activate tf_env
  pip install tf2onnx
  python convert_keras_to_onnx.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

ARTIFACTS_DIR = Path("model_artifacts_cnn")
KERAS_PATH = ARTIFACTS_DIR / "modelo_cnn_hybrid.keras"
ONNX_PATH  = ARTIFACTS_DIR / "modelo_cnn_hybrid.onnx"


def main():
    if not KERAS_PATH.exists():
        print(f"ERROR: No se encontró {KERAS_PATH}")
        return

    if ONNX_PATH.exists():
        print(f"Ya existe {ONNX_PATH}, se sobreescribirá.")

    print(f"Cargando modelo Keras: {KERAS_PATH}")
    import tensorflow as tf
    model = tf.keras.models.load_model(str(KERAS_PATH), compile=False)
    model.summary()

    print(f"\nConvirtiendo a ONNX...")
    import tf2onnx
    import onnxruntime as ort

    # Obtener input spec del modelo
    input_shape = model.input_shape  # (None, 224, 224, 3)
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

    # Guardar
    import onnx
    onnx.save(onnx_model, str(ONNX_PATH))
    print(f"Modelo ONNX guardado en: {ONNX_PATH}")

    # Verificar
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    import numpy as np
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    result = session.run(None, {session.get_inputs()[0].name: dummy})
    print(f"Verificación OK — output shape: {result[0].shape}")
    print(f"\nListo! Ahora puedes usar el benchmark sin TensorFlow.")


if __name__ == "__main__":
    main()
